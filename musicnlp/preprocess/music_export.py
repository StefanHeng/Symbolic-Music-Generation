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
            it = enumerate(filenames)
            if with_tqdm:
                it = tqdm(it, total=len(filenames), desc='Extracting music', unit='song')
            for i_fl, fnm in it:
                if with_tqdm:
                    it.set_postfix(fnm=stem(fnm))
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
        logger = get_logger('Combine Single-Extracted Songs')
        logger.info(f'Combining {logi(len(filenames))} songs... ')

        def load_single(fl):
            with open(fl, 'r') as f_:
                return json.load(f_)
        songs = []
        it = tqdm(filenames, desc='Loading songs', unit='song')
        for fnm in it:
            it.set_postfix(fnm=stem(fnm))
            songs.append(load_single(fnm))

        typ, meta, song = songs[0]['encoding_type'], songs[0]['extractor_meta'], songs[0]['music']
        it = tqdm(songs[1:], desc='Combining songs', unit='song', total=len(songs) - 1)
        it.update(1)
        songs_out = [song]
        for s in it:
            assert s['encoding_type'] == typ and s['extractor_meta'] == meta, 'Metadata for all songs must be the same'
            songs_out.append(s['music'])
        d_out = dict(encoding_type=typ, extractor_meta=meta, music=songs_out)

        meta = MusicExtractor.meta2fnm_meta(meta)
        out_path = f'{output_filename}, n={len(filenames)}, meta={meta}, {now(for_path=True)}'
        out_path = os_join(path_out, f'{out_path}.json')
        logger.info(f'Writing combined songs to {logi(out_path)}... ')
        with open(out_path, 'w') as f:
            json.dump(d_out, f)  # no indent to save disk space
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
        logger = get_logger('JSON=>HF dataset')
        logger.info(f'Loading {logi(fnm)} JSON file... ')
        with open(os_join(path_out, f'{fnm}.json')) as f:
            dset = json.load(f)
        songs, meta = dset['music'], dset['extractor_meta']
        d_info = dict(json_filename=fnm, extractor_meta=meta)

        expected_col_names = {'score', 'title', 'song_path', 'keys'}

        def prep_entry(d: Dict) -> Dict:
            for k in ['warnings', 'duration', 'song_path']:
                del d[k]
            assert all(k in expected_col_names for k in d.keys())  # sanity check
            return d
        entries = []
        it = tqdm(songs, desc='Preparing songs', unit='song')
        for s in it:
            it.set_postfix(fnm=stem(s['song_path']))
            entries.append(prep_entry(s))

        logger.info('Creating HuggingFace dataset... ')
        info = datasets.DatasetInfo(description=json.dumps(d_info))
        dset = datasets.Dataset.from_pandas(pd.DataFrame(entries), info=info)
        if split_args:
            dset = dset.train_test_split(**split_args)

        path = os_join(path_out, 'hf')
        logger.info(f'Saving dataset to {logi(path)}... ')
        os.makedirs(path, exist_ok=True)
        dset.save_to_disk(os_join(path, fnm))
        return dset


if __name__ == '__main__':
    mic.output_width = 512

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
        dnm = 'LMD, MS'
        # dnm = 'LMD, LP'

        # pl_md = 'thread'
        pl_md = 'process'  # seems to be the fastest
        # pl_md = 'thread-in-process'  # ~20% slower

        # mode = 'melody'
        mode = 'full'
        args = dict(
            extractor_args=dict(mode=mode, greedy_tuplet_pitch_threshold=1),
            save_each=True,
            with_tqdm=True,
            # parallel=False,
            parallel=8,
            parallel_mode=pl_md,
            # n_worker=16
        )

        if 'LMD' in dnm:
            # grp_nm = 'many'
            # grp_nm = 'many, lp'
            grp_nm = '060000-070000'
            # grp_nm = '160000-170000'
            # grp_nm = '170000-178561'

            # resume = False
            resume = True
            if resume:
                dir_nm_ = f'22-09-29_LMD_{{md={mode[0]}}}'
            else:
                date = now(fmt='short-date')
                dir_nm_ = f'{date}_LMD_{{md={mode[0]}}}'
            path_out = os_join(music_util.get_processed_path(), 'intermediate', dir_nm_, grp_nm)
            # dnm = 'LMD-cleaned-subset'

            def get_lmd_paths(dir_nm: str) -> List[str]:
                pattern = os_join(u.dset_path, 'converted', dnm, dir_nm, '*.mxl')
                mic(pattern)
                return sorted(glob.iglob(pattern, recursive=True))

            if 'many' in grp_nm:
                paths = sum([get_lmd_paths(d) for d in [
                    # '000000-010000',
                    # '010000-020000',
                    # '020000-030000',
                    # '030000-040000',
                    # '040000-050000',
                    '050000-060000',
                    # '060000-070000',
                    # '070000-080000',
                    # '080000-090000',
                    # '090000-100000',
                    # '100000-110000',
                    # '110000-120000',
                    # '120000-130000',
                    # '130000-140000',
                    # '140000-150000',
                    # '150000-160000',
                    # '160000-170000',
                    # '170000-178561'
                    # grp_nm
                ]], start=[])
            else:
                paths = get_lmd_paths(grp_nm)
            args['filenames'] = paths
        else:
            args['filenames'] = dnm

            # resume = False
            resume = True
            if resume:
                if dnm == 'POP909':
                    dir_nm_ = f'22-09-29_POP909_{{md={mode[0]}}}'
                else:
                    assert dnm == 'MAESTRO'
                    dir_nm_ = f'22-09-29_MAESTRO_{{md={mode[0]}}}'
            else:
                date = now(fmt='short-date')
                dir_nm_ = f'{date}_{dnm}_{{md={mode[0]}}}'
            path_out = os_join(music_util.get_processed_path(), 'intermediate', dir_nm_)
        args['path_out'] = path_out
        me(**args)
    export2json()

    def combine_single_json_songs(singe_song_dir: str, dataset_name: str):
        fl_pattern = '*.json'
        if 'LMD' in dataset_name:
            fl_pattern = f'**/{fl_pattern}'
        fnms = sorted(glob.iglob(os_join(music_util.get_processed_path(), 'intermediate', singe_song_dir, fl_pattern)))
        mic(len(fnms))
        # fnms = fnms[:1024]  # TODO: debugging
        out_fnm = f'{PKG_NM} music extraction, dnm={dataset_name}'
        songs = me.combine_saved_songs(filenames=fnms, output_filename=out_fnm)
        mic(songs.keys(), len(songs['music']))

    def combine():
        md = 'full'
        if md == 'melody':
            # combine_single_json_songs(singe_song_dir='2022-05-19_17-07-40_POP909', dataset_name='POP909')
            # combine_single_json_songs(singe_song_dir='2022-05-19_17-20-29_MAESTRO', dataset_name='MAESTRO')
            combine_single_json_songs(singe_song_dir='2022-05-20_09-39-16_LMD', dataset_name='LMD')
        else:
            # combine_single_json_songs(
            #     singe_song_dir='2022-08-02_17-28-41_POP909, md=f', dataset_name='POP909')
            # combine_single_json_songs(
            #     singe_song_dir='2022-08-02_17-47-08_MAESTRO, md=f', dataset_name='MAESTRO')
            combine_single_json_songs(singe_song_dir='2022-08-02_19-16-56_LMD, md=f', dataset_name='LMD')
    # combine()

    def json2dset_with_split():
        """
        Split the data for now, when amount of data is not huge
        """
        seed = sconfig('random-seed')
        # mode = 'melody'
        mode = 'full'
        if mode == 'melody':
            # fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, ' \
            #       '2022-05-20_14-52-04'
            # fnm = 'musicnlp music extraction, dnm=MAESTRO, n=1276, meta={mode=melody, prec=5, th=1}, ' \
            #       '2022-05-20_14-52-28'
            fnm = 'musicnlp music extraction, dnm=LMD, n=176640, meta={mode=melody, prec=5, th=1}, 2022-05-27_15-23-20'
        else:
            # fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=full, prec=5, th=1}, ' \
            #       '2022-08-02_20-11-17'
            # fnm = 'musicnlp music extraction, dnm=MAESTRO, n=1276, meta={mode=full, prec=5, th=1}, ' \
            #       '2022-08-02_20-12-23'
            fnm = 'musicnlp music extraction, dnm=LMD, n=176640, meta={mode=full, prec=5, th=1}, 2022-09-24_13-26-34'
        dset = me.json2dataset(fnm, split_args=dict(test_size=0.02, shuffle=True, seed=seed))
        mic(dset)
        mic(len(dset['train']), len(dset['test']))
        mic(dset['train'][:2], dset['test'][:2])
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
                    # mic(song.keys())
                    assert 'keys' not in song['music']  # sanity check
                    # mic(fl_nm, song.keys(), song['music'].keys())
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
        # mic(len(fnms))
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
        mic(count_old_key)
    # fix_key_api_change()

    def combine_single_json_songs_with_key():
        # dir_nm = 'POP909 save single 04-10_02.15, add key'
        dir_nm = 'LMD-cleaned_subset save single 04-09_21-51, add key'
        # output_fnm = f'{PKG_NM} music extraction, dnm=POP909'
        output_fnm = f'{PKG_NM} music extraction, dnm=LMD-cleaned-subset'
        fnms = sorted(
            glob.iglob(os_join(music_util.get_processed_path(), 'intermediate', dir_nm, '*.json')))
        songs = me.combine_saved_songs(filenames=fnms, output_filename=output_fnm)
        mic(songs.keys(), len(songs['music']))
    # combine_single_json_songs_with_key()

    def json2dset_with_key():
        fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        dset = me.json2dataset(fnm)
        mic(dset, dset.column_names, dset[:5])
    # json2dset_with_key()

    def check_dset_with_key_features():
        dnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        dset = datasets.load_from_disk(os_join(music_util.get_processed_path(), 'processed', dnm))
        feat_keys = dset.features['keys']
        mic(type(feat_keys))
        mic(dset[:4]['keys'])
    # check_dset_with_key_features()

    def chore_move_proper_folder():
        """
        Batch processing writes to the same folder, move them to sub folders for LMD
        """
        import re
        import shutil

        dir_nm = '22-09-29_LMD_{md=f}'
        path_process_base = os_join(u.dset_path, 'processed', 'intermediate', dir_nm)
        path_to_process = os_join(path_process_base, 'many')
        mic(path_to_process)
        paths = sorted(glob.iglob(os_join(path_to_process, '*.json'), recursive=True))
        pattern = re.compile(r'^Music Export - (?P<ordinal>\d*)$')
        o2f = music_util.Ordinal2Fnm(total=sconfig('datasets.LMD.meta.n_song'), group_size=int(1e4))
        it = tqdm(paths)
        for path in it:
            m = pattern.match(stem(path))
            assert m is not None
            o = int(m.group('ordinal'))

            fnm, dir_nm = o2f(o, return_parts=True)
            fnm = f'Music Export - {fnm}.json'
            it.set_postfix(fnm=f'{dir_nm}/{fnm}')
            path_out = os_join(path_process_base, dir_nm)
            os.makedirs(path_out, exist_ok=True)

            path_out = os_join(path_out, fnm)
            assert not os.path.exists(path_out)
            shutil.move(path, path_out)
    # chore_move_proper_folder()

    def fix_wrong_moved_fnm():
        import shutil
        dir_nm = '2022-05-20_09-39-16_LMD, MS'
        path_process_base = os_join(u.dset_path, 'processed', 'intermediate', dir_nm)
        path_to_process = os_join(path_process_base, '100000-110000')
        paths = sorted(glob.iglob(os_join(path_to_process, '*.json'), recursive=True))
        for path in tqdm(paths):
            fnm = stem(path)[-11:-5]
            path_new = os_join(path_to_process, f'Music Export - {fnm}.json')
            # mic(fnm, path, path_new)
            # exit(1)
            shutil.move(path, path_new)
    # fix_wrong_moved_fnm()

    def sanity_check_export():
        """
        LMD is a large dataset with sub-folders & conversion errors

        Make sure, every valid music file is exported
        """
        meta_fnm = '2022-05-27_14-31-43, LMD conversion meta'
        meta_path = os_join(u.dset_path, 'converted', f'{meta_fnm}.csv')
        df = pd.read_csv(meta_path)

        # n_song = sconfig('datasets.LMD.meta.n_song')
        # o2f = music_util.Ordinal2Fnm(total=n_song, group_size=int(1e4))
        # mic(o2f.total)

        dir_nm = f'2022-05-20_09-39-16_LMD, MS'
        path_exported = os_join(music_util.get_processed_path(), 'intermediate', dir_nm)
        paths_exported = set(glob.iglob(os_join(path_exported, '**/*.json'), recursive=True))
        mic(len(df), len(paths_exported))

        it = tqdm(df.iterrows(), unit='fl', total=len(df))
        for idx, song in it:
            _dir_nm, fnm = song.file_name.split('/')
            fnm = stem(fnm)
            it.set_postfix(fnm=fnm)
            if song.status == 'converted':
                path = os_join(path_exported, _dir_nm, f'Music Export - {fnm}.json')
                # mic(path)
                assert os.path.exists(path)  # a converted file is extracted
                paths_exported.remove(path)
            # mic(idx, song)
            # exit(1)
        mic(paths_exported)
        assert len(paths_exported) == 0  # every exported file is accounted for
    # sanity_check_export()
