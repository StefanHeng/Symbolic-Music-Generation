import glob
from typing import Optional

import datasets

from musicnlp.util import *
import musicnlp.util.music as music_util
from music_extractor import MusicExtractor


class MusicExport:
    """
    Batch export extracted/tokenized music from `MusicTokenizer` in a more accessible format
    """
    def __init__(self, mode='melody', verbose: Union[bool, str] = False):
        """
        :param mode: One of [`melody`, `full`], see `MusicTokenizer`
            TODO: support chords in MusicTokenizer
        :param verbose: Arguments to `MusicExtractor`
        """
        self.verbose = verbose
        self.mode = mode

        self.logger = get_logger('Music Export')

    def __call__(
            self,
            filenames: Union[List[str], str],
            output_filename=f'{PKG_NM} music extraction', path_out=get_processed_path(),
            extractor_args: Dict = None, exp='str_join',
            parallel: Union[bool, int] = False, disable_tqdm: bool = False, save_each: bool = False,
    ):
        """
        Writes encoded files to JSON file

        :param filenames: List of MXL file paths to extract, without `.json` extension;
            or dataset name, see `config.datasets`
        :param output_filename: Export file name
        :param exp: Music extraction output mode, see `MusicTokenizer`
        :param parallel: Whether to parallelize extraction
            If true, a batch size may be specified
        :param save_each: If true, each song is saved into a json file separately
            User is advice to keep each call to `__call__` to a different folder to keep track
            Intended for processing large datasets & saving intermediate processed data,
                instead of keeping all of them in memory
        """
        exp_opns = ['str', 'id', 'str_join']
        if exp not in exp_opns:
            raise ValueError(f'Unexpected export mode - got {logi(exp)}, expect one of {logi(exp_opns)}')
        os.makedirs(path_out, exist_ok=True)

        ext_args = dict(  # `save_memory` so that warnings are for each song
            warn_logger=True, verbose=self.verbose, precision=5, mode='melody',
            greedy_tuplet_pitch_threshold=3**9
        ) | (extractor_args or dict())
        extractor = MusicExtractor(**ext_args)
        self.logger.info(f'Music Extractor created with args: {log_dict(ext_args)}')

        dnm_ = None
        if isinstance(filenames, str):  # Dataset name provided
            dnm_ = filenames
            filenames = music_util.get_cleaned_song_paths(filenames, fmt='mxl')
            # filenames = filenames[3000:]
        self.logger.info(f'Extracting {logi(len(filenames))} songs with {log_dict(dict(save_each=save_each))}... ')

        pbar = None

        def call_single(fl_nm) -> Optional[Dict]:
            if not hasattr(call_single, 'processed_count'):
                call_single.processed_count = 0
            try:
                call_single.processed_count += 1  # Potential data race?
                # Should not exceed 255 limit, see `musicnlp.util.music.py
                fl_nm_single_out = os.path.join(path_out, f'Music Export - {stem(fl_nm)}.json')
                if save_each and os.path.exists(fl_nm_single_out):  # File already processed, ignore
                    if pbar:
                        pbar.update(1)
                    return
                else:
                    ret = extractor(fl_nm, exp=exp, return_meta=True)
                    if save_each:
                        if pbar:
                            pbar.update(1)
                        d_out = dict(encoding_type=exp, extractor_meta=extractor.meta, music=ret, mxl_path=fl_nm)
                        with open(fl_nm_single_out, 'w') as f_:
                            json.dump(d_out, f_, indent=4)
                    else:
                        assert not pbar
                        return ret
            except Exception as e:
                self.logger.error(f'Failed to extract {logi(fl_nm)}, {logi(e)}')  # Abruptly stop the process
                raise ValueError(f'Failed to extract {logi(fl_nm)}')

        if parallel:
            if not disable_tqdm:
                pbar = tqdm(total=len(filenames), desc='Extracting music', unit='song')

            def batched_map(fnms_, s, e):
                return [call_single(fnms_[i]) for i in range(s, e)]
            bsz = (isinstance(parallel, int) and parallel) or 32
            lst_out = batched_conc_map(batched_map, filenames, batch_size=bsz)
            pbar.close()
        else:
            lst_out = []
            gen = enumerate(filenames)
            if not disable_tqdm:
                gen = tqdm(gen, total=len(filenames), desc='Extracting music', unit='song')
            for i_fl, fnm in gen:
                lst_out.append(call_single(fnm))

        if not save_each:
            if dnm_ is not None:
                output_filename += f', dnm={dnm_}'
            meta = MusicExtractor.meta2fnm_meta(extractor.meta)
            output_filename += f', n={len(filenames)}, meta={meta}, {now(for_path=True)}'
            output_filename = os.path.join(path_out, f'{output_filename}.json')
            with open(output_filename, 'w') as f:
                # TODO: Knowing the extracted dict, expand only the first few levels??
                json.dump(dict(encoding_type=exp, extractor_meta=extractor.meta, music=lst_out), f, indent=4)
            self.logger.info(f'Extracted {logi(len(lst_out))} songs written to {logi(output_filename)}')

    @staticmethod
    def combine_saved_songs(
            filenames: List[str], output_filename=f'{PKG_NM} music extraction', path_out=get_processed_path(),
    ) -> Optional[Dict]:
        """
        Combine the individual single-song json file exports to a single file,
            as if running `__call__` with save_each off
        """
        def load_single(fnm):
            with open(fnm, 'r') as f_:
                # ic(fnm)
                return json.load(f_)
        songs = [load_single(fnm) for fnm in filenames]
        typ, meta, song = songs[0]['encoding_type'], songs[0]['extractor_meta'], songs[0]['music']
        songs_out = [song]
        for s in songs[1:]:
            assert s['encoding_type'] == typ and s['extractor_meta'] == meta, 'Metadata for all songs must be the same'
            songs_out.append(s['music'])
        d_out = dict(encoding_type=typ, extractor_meta=meta, music=songs_out)

        meta = MusicExtractor.meta2fnm_meta(meta)
        output_filename = f'{output_filename}, n={len(filenames)}, meta={meta}, {now(for_path=True)}'
        with open(os.path.join(path_out, f'{output_filename}.json'), 'w') as f:
            json.dump(d_out, f)  # no indent saves disk space
        return d_out

    @staticmethod
    def json2dataset(
            fnm: str, path_out=get_processed_path(), split_args: Dict = None
    ) -> Union[datasets.Dataset, datasets.DatasetDict]:
        """
        Save extracted `.json` dataset by `__call__`, as HuggingFace Dataset to disk

        :param fnm: File name to a combined json dataset
        :param path_out: Dataset export path
        :param split_args: arguments for datasets.Dataset.
        """
        with open(os.path.join(path_out, f'{fnm}.json')) as f:
            dset = json.load(f)
        songs, meta = dset['music'], dset['extractor_meta']
        d_info = dict(json_filename=fnm, extractor_meta=meta)

        def prep_entry(d: Dict) -> Dict:
            del d['warnings']
            return d
        dset = datasets.Dataset.from_pandas(
            pd.DataFrame([prep_entry(d) for d in songs]),
            info=datasets.DatasetInfo(description=json.dumps(d_info)),
        )
        if split_args:
            dset = dset.train_test_split(**split_args)
        path = os.path.join(path_out, 'processed')
        os.makedirs(path, exist_ok=True)
        dset.save_to_disk(os.path.join(path, fnm))
        return dset


if __name__ == '__main__':
    from icecream import ic

    ic.lineWrapWidth = 400

    seed = config('random-seed')

    me = MusicExport()
    # me = MusicExport(verbose=True)
    # me = MusicExport(verbose='single')

    def check_sequential():
        # me('LMD-cleaned-subset', parallel=False)
        me('LMD-cleaned-subset', parallel=False, extractor_args=dict(greedy_tuplet_pitch_threshold=1))
    # check_sequential()
    # profile_runtime(check_sequential)

    def check_parallel():
        me('LMD-cleaned-subset', parallel=3)
    # check_parallel()

    def check_lower_threshold():
        # th = 4**5  # this number we can fairly justify
        th = 1  # The most aggressive, for the best speed, not sure about losing quality
        me('LMD-cleaned-subset', parallel=3, extractor_args=dict(greedy_tuplet_pitch_threshold=th))
    # check_lower_threshold()

    def export2json():
        dnm = 'POP909'
        # dnm = 'LMD-cleaned-subset'
        # me(dnm)
        me(dnm, parallel=32, extractor_args=dict(greedy_tuplet_pitch_threshold=1))
    # export2json()

    def export2json_save_each(
            filenames: Union[str, List[str]] = 'LMD-cleaned-subset',
            save_dir: str = 'LMD-cleaned_subset save single 04-09_21-51'
    ):
        path_out = os.path.join(get_processed_path(), 'intermediate', save_dir)
        # parallel = 3
        parallel = 64
        me(
            filenames, parallel=False, extractor_args=dict(greedy_tuplet_pitch_threshold=1),
            path_out=path_out, save_each=True
        )
    # export2json_save_each()
    # export2json_save_each(filenames='POP909', save_dir='POP909 save single 04-10_02.15')
    # export2json_save_each(filenames=music_util.get_cleaned_song_paths('LMD-cleaned-subset', fmt='mxl')[3000:])

    def combine_single_json_songs(singe_song_dir: str, output_fnm: str):
        fnms = sorted(glob.iglob(os.path.join(get_processed_path(), 'intermediate', singe_song_dir, '*.json')))
        songs = me.combine_saved_songs(filenames=fnms, output_filename=output_fnm)
        ic(songs.keys(), len(songs['music']))
    # combine_single_json_songs(
    #     singe_song_dir='POP909 save single 04-10_02.15',
    #     output_fnm=f'{PKG_NM} music extraction, dnm=POP909'
    # )
    # combine_single_json_songs(
    #     singe_song_dir='LMD-cleaned_subset save single 04-09_21-51',
    #     output_fnm=f'{PKG_NM} music extraction, dnm=LMD-cleaned-subset'
    # )

    def json2dset():
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-22 19-00-40'
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-25 20-59-06'
        fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-03-01 02-29-29'
        dset = me.json2dataset(fnm)
        ic(dset, dset[:5])
    # json2dset()

    def json2dset_with_split():
        """
        Split the data for now, when amount of data is not huge
        """
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-51-01'
        # fnm = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
        #       'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-52-41'
        fnm = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
              'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-10_19-49-52'
        # for 10k data in the LMD-cleaned dataset, this is like 200 songs, should be good enough
        dset = me.json2dataset(fnm, split_args=dict(test_size=0.02, shuffle=True, seed=seed))
        ic(dset)
        ic(len(dset['train']), len(dset['test']))
        ic(dset['train'][:3], dset['test'][:3])
    json2dset_with_split()
