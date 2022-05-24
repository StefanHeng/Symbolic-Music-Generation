import os
import glob
import math
import json
import logging
import datetime
from os.path import join as os_join
from typing import List, Tuple, Dict, Optional, Iterable, Union

import pandas as pd
import datasets
from tqdm import tqdm

from stefutil import *
from musicnlp.util import *
import musicnlp.util.music as music_util
from music_extractor import MusicExtractor


class SingleExport:
    """
    Class instead of local function for multiprocessing

    TODO: then my entire extraction needs to be pickleable...
    """
    def __init__(
            self,
            output_path: str, save_each: bool, logger: logging.Logger,
            extractor: MusicExtractor, exp: str, log2console: bool = True,
            halt_on_error: bool = True
    ):
        self.output_path = output_path
        self.save_each = save_each
        self.log2console = log2console
        self.logger = logger
        self.extractor = extractor
        self.exp = exp
        self.halt_on_error = halt_on_error

        self.processed_count = 0

    def __call__(self, fl_nm) -> Optional[Dict]:
        try:
            self.processed_count += 1  # Potential data race?
            # Should not exceed 255 limit, see `musicnlp.util.music.py
            fnm_ = stem(fl_nm)
            fl_nm_single_out = os_join(self.output_path, f'Music Export - {fnm_}.json')
            if self.save_each and os.path.exists(fl_nm_single_out):  # File already processed, ignore
                if self.log2console:
                    self.logger.info(f'{logi(self.processed_count)}:{logi(fnm_)} already processed, skipping...')
                return
            else:
                ret = self.extractor(fl_nm, exp=self.exp, return_meta=True, return_key=True)
                if self.log2console:
                    self.logger.info(f'{logi(self.processed_count)}:{logi(fnm_)} processing finished ')
                if self.save_each:
                    d_out = dict(encoding_type=self.exp, extractor_meta=self.extractor.meta, music=ret)
                    with open(fl_nm_single_out, 'w') as f_:
                        json.dump(d_out, f_, indent=4)
                else:
                    return ret
        except Exception as e:
            self.logger.error(f'Failed to extract {logi(fl_nm)}, {logi(e)}')  # Abruptly stop the process
            if self.halt_on_error:
                raise ValueError(f'Failed to extract {logi(fl_nm)}')
            else:
                return  # Keep the extraction going


class ThreadedBatchedExport:
    """
    Intended for `parallel_mode` `thread-in-process`, i.e. multiple-threads in a single process to save IO time
    """
    def __init__(self, fn):
        self.single_export = fn

    def __call__(self, filenames: Union[List[str], Tuple[str]]) -> Iterable[Optional[Dict]]:
        return list(conc_map(self.single_export, filenames, with_tqdm=False, mode='thread'))


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
            output_filename=f'{PKG_NM} music extraction', path_out=music_util.get_processed_path(),
            extractor_args: Dict = None, exp='str_join',
            save_each: bool = False, with_tqdm: Union[bool, Dict] = False,
            parallel: Union[bool, int] = False, parallel_mode: str = 'thread', n_worker: int = os.cpu_count()
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
        strt = datetime.datetime.now()
        ca.check_mismatch('Music Extraction Export Type', exp, accepted_values=['str', 'id', 'str_join'])
        ca.check_mismatch('Music Extraction Parallel Mode', parallel_mode, accepted_values=[
            'thread', 'process', 'thread-in-process'
        ])
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
            filenames = music_util.get_converted_song_paths(filenames, fmt='mxl')
            # filenames = filenames[:16]  # TODO: Debugging
        d_log = dict(save_each=save_each, with_tqdm=with_tqdm, parallel=parallel, parallel_mode=parallel_mode)
        n_song = len(filenames)
        self.logger.info(f'Extracting {logi(n_song)} songs with {log_dict(d_log)}... ')

        pbar = None

        log2console = not with_tqdm  # TODO: not working when multiprocessing
        halt_on_error = not bool(parallel)
        ic(halt_on_error)
        export_single = SingleExport(
            path_out, save_each, self.logger, extractor, exp, log2console=log2console, halt_on_error=halt_on_error
        )

        if parallel:
            bsz = (isinstance(parallel, int) and parallel) or 32
            tqdm_args = dict(desc='Extracting music', total=n_song)
            if parallel_mode == 'thread':  # able to log on element-level
                if with_tqdm:
                    tqdm_args.update(dict(unit='song'))
                    pbar = tqdm(**tqdm_args)
                lst_out = batched_conc_map(
                    export_single, filenames, batch_size=bsz, with_tqdm=pbar, mode=parallel_mode, n_worker=n_worker
                )
                if pbar:
                    pbar.close()
            elif parallel_mode == 'process':  # only able to log at batch-level
                if with_tqdm:
                    if isinstance(with_tqdm, dict):
                        tqdm_args.update(with_tqdm)
                    tqdm_args.update(dict(unit='song', chunksize=bsz))
                # don't have to go through my own batched map
                lst_out = conc_map(
                    export_single, filenames, with_tqdm=tqdm_args, mode=parallel_mode, n_worker=n_worker
                )
            else:
                assert parallel_mode == 'thread-in-process'
                total = math.ceil(n_song / bsz)
                tqdm_args.update(dict(unit='ba', total=total))
                it = list(group_n(filenames, bsz))
                fn = ThreadedBatchedExport(export_single)
                lst_out = conc_map(fn, it, with_tqdm=tqdm_args, mode='process', n_worker=n_worker)
        else:
            lst_out = []
            gen = enumerate(filenames)
            if with_tqdm:
                gen = tqdm(gen, total=len(filenames), desc='Extracting music', unit='song')
            for i_fl, fnm in gen:
                lst_out.append(export_single(fnm))

        if not save_each:
            out_fnm = output_filename
            if dnm_ is not None:
                out_fnm += f', dnm={dnm_}'
            meta = MusicExtractor.meta2fnm_meta(extractor.meta)
            out_fnm += f', n={len(filenames)}, meta={meta}, {now(for_path=True)}'
            out_fnm = os_join(path_out, f'{out_fnm}.json')
            with open(out_fnm, 'w') as f:
                # TODO: Knowing the extracted dict, expand only the first few levels??
                json.dump(dict(encoding_type=exp, extractor_meta=extractor.meta, music=lst_out), f, indent=4)
            self.logger.info(f'Extracted {logi(len(lst_out))} songs written to {logi(out_fnm)}')
        self.logger.info(f'Extraction finished in {logi(fmt_delta(datetime.datetime.now() - strt))}')

    @staticmethod
    def combine_saved_songs(
            filenames: List[str], output_filename=f'{PKG_NM} music extraction',
            path_out=music_util.get_processed_path(),
    ) -> Optional[Dict]:
        """
        Combine the individual single-song json file exports to a single file,
            as if running `__call__` with save_each off
        """
        def load_single(fnm):
            with open(fnm, 'r') as f_:
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
        with open(os_join(path_out, f'{output_filename}.json'), 'w') as f:
            json.dump(d_out, f)  # no indent saves disk space
        return d_out

    @staticmethod
    def json2dataset(
            fnm: str, path_out=music_util.get_processed_path(), split_args: Dict = None
    ) -> Union[datasets.Dataset, datasets.DatasetDict]:
        """
        Save extracted `.json` dataset by `__call__`, as HuggingFace Dataset to disk

        :param fnm: File name to a combined json dataset
        :param path_out: Dataset export path
        :param split_args: arguments for datasets.Dataset.
        """
        with open(os_join(path_out, f'{fnm}.json')) as f:
            dset = json.load(f)
        songs, meta = dset['music'], dset['extractor_meta']
        d_info = dict(json_filename=fnm, extractor_meta=meta)

        expected_col_names = {'score', 'title', 'song_path', 'keys'}

        def prep_entry(d: Dict) -> Dict:
            del d['warnings']
            del d['duration']
            assert all(k in expected_col_names for k in d.keys())  # sanity check
            return d
        dset = datasets.Dataset.from_pandas(
            pd.DataFrame([prep_entry(d) for d in songs]),
            info=datasets.DatasetInfo(description=json.dumps(d_info)),
        )
        if split_args:
            dset = dset.train_test_split(**split_args)
        path = os_join(path_out, 'hf')
        os.makedirs(path, exist_ok=True)
        dset.save_to_disk(os_join(path, fnm))
        return dset


if __name__ == '__main__':
    from icecream import ic

    ic.lineWrapWidth = 400

    seed = sconfig('random-seed')

    me = MusicExport()
    # me = MusicExport(verbose=True)
    # me = MusicExport(verbose='single')

    def check_sequential():
        # dnm = 'LMD-cleaned-subset'
        dnm = 'POP909'
        # dnm = 'MAESTRO'
        # me(dnm, parallel=False)
        me(dnm, parallel=False, extractor_args=dict(greedy_tuplet_pitch_threshold=1))
    # check_sequential()
    # profile_runtime(check_sequential)

    def check_parallel():
        me('LMD-cleaned-subset', parallel=3)
    # check_parallel()

    def check_lower_threshold():
        # th = 4**5  # this number we can fairly justify
        th = 1  # The most aggressive, for the best speed, not sure about losing quality
        dnm = 'POP909'
        # dnm = 'LMD-cleaned-subset'
        # pl_md = 'thread'
        # pl_md = 'process'
        pl_md = 'thread-in-process'
        me(dnm, parallel=8, extractor_args=dict(greedy_tuplet_pitch_threshold=th), with_tqdm=True, parallel_mode=pl_md)
    # check_lower_threshold()

    def check_export_json_error():
        dnm = 'MAESTRO'
        dir_nm = sconfig(f'datasets.{dnm}.converted.dir_nm')
        dir_nm = f'{dir_nm}, MS'
        # fnm = 'Franz Liszt - Après Une Lecture De Dante: Fantasia Quasi Sonata, S.161, No. 7.mxl'
        # fnm = 'Johann Sebastian Bach - Prelude And Fugue In E Major, Wtc I, Bwv 854.mxl'
        fnm = 'Franz Liszt - Études D\'exécution Transcendante, No. 5, "feux Follets" S.139:5.mxl'
        path = os_join(u.dset_path, dir_nm, fnm)
        me([path], extractor_args=dict(greedy_tuplet_pitch_threshold=1))
    # check_export_json_error()

    def export2json():
        # dnm = 'POP909'
        # dnm = 'MAESTRO'
        # dnm = 'LMD, MS/000000-010000'
        dnm = 'LMD, MS'
        # dir_nm_ = f'{now(for_path=True)}_{dnm}'
        grp_nm = '070000-080000'
        dir_nm_ = f'2022-05-20_09-39-16_LMD, MS/{grp_nm}'
        path_out = os_join(music_util.get_processed_path(), 'intermediate', dir_nm_)
        # dnm = 'LMD-cleaned-subset'
        # me(dnm)
        # pl_md = 'thread'
        pl_md = 'process'  # seems to be the fastest
        # pl_md = 'thread-in-process'  # ~20% slower
        args = dict(greedy_tuplet_pitch_threshold=1)

        def get_lmd_paths(dir_nm: str) -> List[str]:
            pattern = os_join(u.dset_path, 'converted', dnm, dir_nm, '*.mxl')
            ic(pattern)
            return sorted(glob.iglob(pattern, recursive=True))
        paths = sum([get_lmd_paths(d) for d in [
            # '000000-010000',
            # '010000-020000',
            # '020000-030000',
            # '030000-040000',
            # '040000-050000',
            # '050000-060000',
            # '060000-070000',
            grp_nm
        ]], start=[])
        ic(len(paths))
        me(
            # dnm,
            paths,
            extractor_args=args, path_out=path_out, save_each=True,
            parallel=128,
            with_tqdm=True, parallel_mode=pl_md,
            # n_worker=16
        )
    export2json()

    def export2json_save_each(
            filenames: Union[str, List[str]] = 'LMD-cleaned-subset',
            save_dir: str = 'LMD-cleaned_subset save single 04-09_21-51'
    ):
        path_out = os_join(music_util.get_processed_path(), 'intermediate', save_dir)
        # parallel = 3
        parallel = 64
        me(
            filenames, parallel=False, extractor_args=dict(greedy_tuplet_pitch_threshold=1),
            path_out=path_out, save_each=True
        )
    # export2json_save_each()
    # export2json_save_each(filenames='POP909', save_dir='POP909 save single 04-10_02.15')
    # export2json_save_each(filenames=music_util.get_cleaned_song_paths('LMD-cleaned-subset', fmt='mxl')[3000:])

    def combine_single_json_songs(singe_song_dir: str, dataset_name: str):
        fnms = sorted(glob.iglob(os_join(music_util.get_processed_path(), 'intermediate', singe_song_dir, '*.json')))
        out_fnm = f'{PKG_NM} music extraction, dnm={dataset_name}'
        songs = me.combine_saved_songs(filenames=fnms, output_filename=out_fnm)
        ic(songs.keys(), len(songs['music']))
    # combine_single_json_songs(singe_song_dir='POP909 save single 04-10_02.15', dataset_name='POP909')
    # combine_single_json_songs(
    #     singe_song_dir='LMD-cleaned_subset save single 04-09_21-51',
    #     dataset_name='LMD-cleaned_subset'
    # )
    # combine_single_json_songs(singe_song_dir='2022-05-19_17-07-40_POP909', dataset_name='POP909')
    # combine_single_json_songs(singe_song_dir='2022-05-19_17-20-29_MAESTRO', dataset_name='MAESTRO')

    def json2dset():
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-22 19-00-40'
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-25 20-59-06'
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-03-01 02-29-29'
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-04'
        fnm = 'musicnlp music extraction, dnm=MAESTRO, n=1276, meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-28'
        dset = me.json2dataset(fnm)
        ic(dset, dset[:5])
    # json2dset()

    def json2dset_with_split():
        """
        Split the data for now, when amount of data is not huge
        """
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-51-01'
        # fnm = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
        #       'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-10_19-49-52'
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        # fnm = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
        #       'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-17_11-52-15'
        # for 10k data in the LMD-cleaned subset dataset, this is like 200 songs, should be good enough
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-04'
        fnm = 'musicnlp music extraction, dnm=MAESTRO, n=1276, meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-28'
        dset = me.json2dataset(fnm, split_args=dict(test_size=0.02, shuffle=True, seed=seed))
        ic(dset)
        ic(len(dset['train']), len(dset['test']))
        ic(dset['train'][:3], dset['test'][:3])
    # json2dset_with_split()

    def fix_insert_key():
        """
        As extracting from all the 10k songs is time-consuming,
        temporarily insert keys into the already processed json files
        """
        from musicnlp.preprocess.key_finder import KeyFinder

        # dnm = 'POP909'
        dnm = 'LMD-cleaned-subset'
        # dir_nm = 'POP909 save single 04-10_02.15'
        dir_nm = 'LMD-cleaned_subset save single 04-09_21-51'
        dir_nm_out = f'{dir_nm}, add key'
        path = os_join(music_util.get_processed_path(), 'intermediate', dir_nm)
        path_out = os_join(music_util.get_processed_path(), 'intermediate', dir_nm_out)
        os.makedirs(path_out, exist_ok=True)
        fnms = sorted(glob.iglob(os_join(path, '*.json')))

        nm = 'Insert Key Back'
        logger = get_logger(nm)
        pbar = tqdm(total=len(fnms), desc=nm, unit='song')

        dir_nm_dset = 'LMD-cleaned_valid' if dnm == 'LMD-cleaned-subset' else dnm  # for pop909

        def song_title2path(title: str) -> str:
            # Needed cos the original json files may not be processed on my local computer
            return os_join(BASE_PATH, DSET_DIR, dir_nm_dset, f'{title}.mxl')

        def call_single(fl_nm: str):
            try:
                fnm_out = os_join(path_out, f'{stem(fl_nm)}.json')
                if not os.path.exists(fnm_out):
                    with open(fl_nm, 'r') as f:
                        song = json.load(f)
                    # ic(song.keys())
                    assert 'keys' not in song['music']  # sanity check
                    # ic(fl_nm, song.keys(), song['music'].keys())
                    path_mxl = song_title2path(song['music']['title'])
                    song['music']['keys'] = keys = KeyFinder(path_mxl).find_key(return_type='dict')
                    assert len(keys) > 0
                    with open(fnm_out, 'w') as f:
                        json.dump(song, f, indent=4)
                pbar.update(1)
            except Exception as e:
                logger.error(f'Failed to find key for {logi(fl_nm)}, {logi(e)}')  # Abruptly stop the process
                raise ValueError(f'Failed to find key for {logi(fl_nm)}')
        # for fnm in fnms[:20]:
        #     call_single(fnm)

        def batched_map(fnms_, s, e):
            return [call_single(fnms_[i]) for i in range(s, e)]
        n_worker = os.cpu_count() * 2  # majority of the time is m21 parsing file
        batched_conc_map(batched_map, fnms, batch_size=32, n_worker=n_worker)
    # fix_insert_key()
    # profile_runtime(fix_insert_key)

    def fix_key_api_change():
        """
        Original KeyFinder results are stored in a key `key`, the new API uses `keys`, so update the written files
        """
        dir_nm = 'LMD-cleaned_subset save single 04-09_21-51, add key'
        # dir_nm = 'POP909 save single 04-10_02.15, add key'
        path = os_join(music_util.get_processed_path(), 'intermediate', dir_nm)
        fnms = sorted(glob.iglob(os_join(path, '*.json')))
        # ic(len(fnms))
        # exit(1)

        count_old_key = 0
        for fnm in tqdm(fnms):
            with open(fnm, 'r') as f:
                d = json.load(f)
            if 'key' in d['music']:
                d['music']['keys'] = d['music'].pop('key')
                count_old_key += 1
                with open(fnm, 'w') as f:  # Override the original file
                    json.dump(d, f, indent=4)
        ic(count_old_key)
    # fix_key_api_change()

    def combine_single_json_songs_with_key():
        # dir_nm = 'POP909 save single 04-10_02.15, add key'
        dir_nm = 'LMD-cleaned_subset save single 04-09_21-51, add key'
        # output_fnm = f'{PKG_NM} music extraction, dnm=POP909'
        output_fnm = f'{PKG_NM} music extraction, dnm=LMD-cleaned-subset'
        fnms = sorted(
            glob.iglob(os_join(music_util.get_processed_path(), 'intermediate', dir_nm, '*.json')))
        songs = me.combine_saved_songs(filenames=fnms, output_filename=output_fnm)
        ic(songs.keys(), len(songs['music']))
    # combine_single_json_songs_with_key()

    def json2dset_with_key():
        fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        dset = me.json2dataset(fnm)
        ic(dset, dset.column_names, dset[:5])
    # json2dset_with_key()

    def check_dset_with_key_features():
        dnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        dset = datasets.load_from_disk(os_join(music_util.get_processed_path(), 'processed', dnm))
        feat_keys = dset.features['keys']
        ic(type(feat_keys))
        ic(dset[:4]['keys'])
    # check_dset_with_key_features()
