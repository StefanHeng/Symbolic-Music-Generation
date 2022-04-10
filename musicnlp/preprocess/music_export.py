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
            # fnms = music_util.get_cleaned_song_paths(fnms, fmt='mxl')[:40]
            filenames = music_util.get_cleaned_song_paths(filenames, fmt='mxl')[6000:]
        self.logger.info(f'Extracting {logi(len(filenames))} songs with {log_dict(dict(save_each=save_each))}... ')

        pbar = None

        def call_single(fl_nm) -> Optional[Dict]:
            if not hasattr(call_single, 'processed_count'):
                call_single.processed_count = 0
            try:
                call_single.processed_count += 1  # Potential data race?
                fl_nm_out = None
                if save_each:
                    # Should not exceed 255 limit, see `musicnlp.util.music.py
                    fl_nm_out = os.path.join(path_out, f'Music Export - {stem(fl_nm)}.json')
                    if os.path.exists(fl_nm_out):  # File already processed, ignore
                        return
                ret = extractor(fl_nm, exp=exp, return_meta=True)
                if pbar:
                    pbar.update(1)
                    d_out = dict(encoding_type=exp, extractor_meta=extractor.meta, music=ret, mxl_path=fl_nm)
                    with open(fl_nm_out, 'w') as f_:
                        json.dump(d_out, f_, indent=4)
                else:
                    return ret
            except Exception as e:
                self.logger.error(f'Failed to extract {logi(fl_nm)}, {logi(e)}')  # Abruptly stop the process
                raise ValueError(f'Failed to extract {logi(fl_nm)}')

        if parallel:
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
            json.dump(d_out, f, indent=4)
        return d_out

    @staticmethod
    def json2dataset(fnm: str, path_out=get_processed_path()) -> datasets.Dataset:
        """
        Save extracted `.json` dataset by `__call__`, as HuggingFace Dataset to disk
        """
        with open(os.path.join(path_out, f'{fnm}.json')) as f:
            dset_: Dict = json.load(f)
        tr = dset_['music']  # TODO: All data as training?

        def prep_entry(d: Dict) -> Dict:
            del d['warnings']
            return d
        dset = datasets.Dataset.from_pandas(
            pd.DataFrame([prep_entry(d) for d in tr]),
            info=datasets.DatasetInfo(description=json.dumps(dict(precision=dset_['precision'])))
        )
        dset.save_to_disk(os.path.join(path_out, 'hf_datasets', fnm))
        return dset


if __name__ == '__main__':
    from icecream import ic

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
        # dnm = 'POP909'
        dnm = 'LMD-cleaned-subset'
        # me(dnm)
        me(dnm, parallel=32, extractor_args=dict(greedy_tuplet_pitch_threshold=1))
    # export2json()

    def export2json_save_each():
        path_out = os.path.join(get_processed_path(), '04-09_21-51')
        # parallel = 3
        parallel = 64
        me(
            'LMD-cleaned-subset', parallel=parallel, extractor_args=dict(greedy_tuplet_pitch_threshold=1),
            path_out=path_out, save_each=True
        )
    # export2json_save_each()

    def json2dset():
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-22 19-00-40'
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-25 20-59-06'
        fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-03-01 02-29-29'
        dset = me.json2dataset(fnm)
        ic(dset, dset[:5])
    # json2dset()

    def combine_single_json_songs():
        fnms = sorted(glob.iglob(os.path.join(get_processed_path(), '04-09_21-51', '*.json')))
        songs = me.combine_saved_songs(fnms)
        ic(songs.keys(), len(songs['music']))
    combine_single_json_songs()
