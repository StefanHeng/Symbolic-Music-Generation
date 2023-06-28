import os
import glob
import math
import json
import logging
import datetime
from os.path import join as os_join
from typing import List, Tuple, Dict, Optional, Iterable, Union, Any
from dataclasses import asdict

import pandas as pd
import datasets
from unidecode import unidecode
from tqdm import tqdm

from stefutil import *
from musicnlp.util import *
import musicnlp.util.music as music_util
from musicnlp.preprocess.music_extractor import MusicExtractorOutput, MusicExtractor


__all__ = ['MusicExport']


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
                    self.logger.info(f'{pl.i(self.processed_count)}:{pl.i(fnm_)} already processed, skipping...')
                return
            else:
                ret_: MusicExtractorOutput = self.extractor(fl_nm, exp=self.exp, return_meta=True, return_key=True)
                ret = asdict(ret_)
                if self.log2console:
                    self.logger.info(f'{pl.i(self.processed_count)}:{pl.i(fnm_)} processing finished ')
                if self.save_each:
                    d_out = dict(encoding_type=self.exp, extractor_meta=self.extractor.meta, music=ret)
                    with open(fl_nm_single_out, 'w') as f_:
                        json.dump(d_out, f_, indent=4)
                else:
                    return ret
        except Exception as e:
            self.logger.error(f'Failed to extract {pl.i(fl_nm)}, {pl.i(e)}')  # Abruptly stop the process
            if self.halt_on_error:
                raise ValueError(f'Failed to extract {pl.i(fl_nm)}')
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
            filenames: Union[List[str], str], dataset_name2songs_args: Optional[Dict] = None,
            output_filename=f'{PKG_NM} music extraction', path_out=music_util.get_processed_path(),
            extractor_args: Dict = None, exp='str_join',
            save_each: bool = False, with_tqdm: Union[bool, Dict] = False,
            parallel: Union[bool, int] = False, parallel_mode: str = 'thread', n_worker: int = os.cpu_count()
    ):
        """
        Writes encoded files to JSON file

        :param filenames: List of MXL file paths to extract, without `.json` extension;
            or dataset name, see `config.datasets`
        :param dataset_name2songs_args: args to `get_converted_song_paths`
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
        self.logger.info(f'Music Extractor created with args: {pl.i(ext_args)}')

        dnm_ = None
        if isinstance(filenames, str):  # Dataset name provided
            dnm_ = filenames
            args = dict(fmt='mxl') | (dataset_name2songs_args or dict())
            filenames = music_util.get_converted_song_paths(dataset_name=dnm_, **args)
            # filenames = filenames[:16]  # TODO: Debugging
        d_log = dict(save_each=save_each, with_tqdm=with_tqdm, parallel=parallel, parallel_mode=parallel_mode)
        n_song = len(filenames)
        self.logger.info(f'Extracting {pl.i(n_song)} songs with {pl.i(d_log)}... ')

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
                if with_tqdm is True:
                    with_tqdm = dict()
                if save_each:
                    tqdm_args |= dict(unit='song') | (with_tqdm or dict())
                    args = dict(with_tqdm=tqdm_args, mode=parallel_mode, n_worker=n_worker, batch_size=bsz)
                    for _ in conc_yield(fn=export_single, args=filenames, **args):  # No need to create list
                        pass
                else:
                    tqdm_args |= dict(unit='song', chunksize=bsz) | (with_tqdm or dict())
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
                    it.set_postfix(fnm=pl.i(stem(fnm)))
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
            self.logger.info(f'Extracted {pl.i(len(lst_out))} songs written to {pl.i(out_fnm)}')
        self.logger.info(f'Extraction finished in {pl.i(fmt_delta(datetime.datetime.now() - strt))}')

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
        logger.info(f'Combining {pl.i(len(filenames))} songs... ')

        def load_single(fl):
            with open(fl, 'r') as f_:
                return json.load(f_)
        songs = []
        it = tqdm(filenames, desc='Loading songs', unit='song')
        for fnm in it:
            it.set_postfix(fnm=pl.i(stem(fnm)))
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
        date = now(fmt='short-date')
        out_path = f'{date}_{output_filename}_{{n={len(filenames)}}}_{meta}'
        out_path = os_join(path_out, f'{out_path}.json')
        logger.info(f'Writing combined songs to {pl.i(out_path)}... ')
        with open(out_path, 'w') as f:
            json.dump(d_out, f)  # no indent to save disk space
        return d_out

    @staticmethod
    def json2dataset(
            fnm: str, path_out=music_util.get_processed_path(), split: bool = False, test_size: float = 0.02,
            test_size_range: Tuple[int, int] = None, seed: int = None,
            pre_determined_split: Dict[str, Dict[str, str]] = None
    ) -> Union[datasets.Dataset, datasets.DatasetDict]:
        """
        Save extracted `.json` dataset by `__call__`, as HuggingFace Dataset to disk

        :param fnm: File name to a combined json dataset
        :param path_out: Dataset export path
        :param split: Whether to split the dataset into train/val/test
        :param test_size: Test set size, as a fraction of the total dataset
        :param test_size_range: Test set size threshold in (min, max)
            will override `test_size` if outside threshold
        :param seed: Random seed for splitting
        :param pre_determined_split: Pre-determined split for each song by piece title
            See `musicnlp.util.music::clean_dataset_paths`
        """
        logger = get_logger('JSON=>HF dataset')
        logger.info(f'Loading {pl.i(fnm)} JSON file... ')
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
            it.set_postfix(fnm=pl.i(stem(s['song_path'])))
            entries.append(prep_entry(s))

        logger.info('Creating HuggingFace dataset... ')
        info = datasets.DatasetInfo(description=json.dumps(d_info))
        dset = datasets.Dataset.from_pandas(pd.DataFrame(entries), info=info)
        if split:
            if pre_determined_split:
                logger.info('Using pre-determined split... ')
                # mic(dset, pre_determined_split)
                # mic(dset, dset[:10]['title'])
                # original_all_titles = sorted(dset[:]['title'])

                # Decoding is needed to drop accents, for weird edge cases TODO, an example below
                #   `César Franck - Prel.:chor.:fug.` and `César Franck - Prel.:chor.:fug.` are different
                #       for the 2nd char, `e w/ accent`, becomes 2 different characters: just the `e` and the `accent`
                splits = ['train', 'validation', 'test']
                split2ttl = {
                    split: set(unidecode(ttl) for ttl, d in pre_determined_split.items() if d['split'] == split)
                    for split in splits
                }
                # mic({split: len(ttls) for split, ttls in split2ttl.items()})
                # mic(split2ttl)
                # tr = dset.filter(lambda x: unidecode(x['title']) in split2ttl['train'])
                # # mic(len(tr))
                # val = dset.filter(lambda x: unidecode(x['title']) in split2ttl['validation'])
                # ts = dset.filter(lambda x: unidecode(x['title']) in split2ttl['test'])
                # dset = datasets.DatasetDict(train=tr, validation=val, test=ts)
                n = len(dset)
                dset = {s: dset.filter(lambda x: unidecode(x['title']) in split2ttl[s]) for s in splits}
                dset = datasets.DatasetDict(**dset)

                mic(dset)
                titles = set().union(*[d[:]['title'] for d in dset.values()])
                assert sum(len(d) for d in dset.values()) == n and len(titles) == n  # sanity check partition

                # all_titles = sorted(set().union(*split2ttl.values()))
                # mic(len(all_titles), all_titles[:10])
                # # titles_not_in_any =
                # for i, (ori, cur) in enumerate(zip(original_all_titles, all_titles)):
                #     if ori != cur:
                #         mic(i, ori, cur)
                # # mic(len(set(original_all_titles).intersection(set(all_titles))))
                #
                # original_all_titles = [unidecode(ttl) for ttl in original_all_titles]
                # all_titles = [unidecode(ttl) for ttl in all_titles]
                #
                # diff1 = sorted(set(original_all_titles) - set(all_titles))
                # diff2 = sorted(set(all_titles) - set(original_all_titles))
                # mic(diff1[:10], diff2[:10])
                #
                # s1 =
                # mic(len(s1), len(s2))
                # for i, (c1, c2) in enumerate(zip(s1, s2)):
                #     if c1 != c2:
                #         mic(i, c1, c2, ord(c1), ord(c2))
                # # enc ='ascii'
                # enc = 'utf-8'
                # s1 = s1.encode(enc, 'replace')
                # s2 = s2.encode(enc, 'replace')
                # mic(s1, s2, len(s1), len(s2))

                raise NotImplementedError
                # met
                # dset = dset.train_test_split(
                #     test_size=None, shuffle=False, seed=seed, train_size=0.8, test_size_from_train=0.5,
                #     train_transform=partial(clean_dataset_paths, pre_determined_split=pre_determined_split)
                # )
                # raise NotImplementedError

            n_ts = len(dset) * test_size

            if test_size_range is not None:
                mi, ma = test_size_range
                n_ts = min(max(n_ts, mi), ma)
            dset = dset.train_test_split(test_size=n_ts, shuffle=True, seed=seed)

        path = os_join(path_out, 'hf')
        logger.info(f'Saving dataset to {pl.i(path)}... ')
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

    def export2json():
        # dnm = 'POP909'
        dnm = 'MAESTRO'
        # dnm = 'LMD, MS'
        # dnm = 'LMD, LP'
        # dnm = 'LMD-cleaned-subset'
        # dnm = 'LMCI, MS'
        # dnm = 'LMCI, LP'
        # dnm = 'NES-MDB'

        # mode = 'melody'
        mode = 'full'
        args: Dict[str, Any] = dict(
            extractor_args=dict(mode=mode, greedy_tuplet_pitch_threshold=1, with_pitch_step=True),
            save_each=True,
            with_tqdm=True
        )
        # args['parallel'] = False
        args['parallel'] = 8
        if args['parallel']:
            # pl_md = 'thread'
            pl_md = 'process'  # seems to be the fastest
            # pl_md = 'thread-in-process'  # ~20% slower
            args.update(dict(parallel_mode=pl_md, n_worker=None))

        dset_path = os_join(get_base_path(), u.dset_dir)

        def get_nested_paths(dir_nm: str) -> List[str]:
            pattern = os_join(dset_path, 'converted', dnm, dir_nm, '*.mxl')
            mic(pattern)
            return sorted(glob.iglob(pattern, recursive=True))

        if 'LMD, ' in dnm:
            # grp_nm = 'many'
            # grp_nm = 'many, lp'
            # grp_nm = '090000-100000'
            # grp_nm = '160000-170000'
            grp_nm = '170000-178561'

            # resume = False
            resume = True
            if resume:
                dir_nm_ = f'23-04-06_LMD_{{md={mode[0]}}}'
            else:
                date = now(fmt='short-date')
                dir_nm_ = f'{date}_LMD_{{md={mode[0]}}}'
            path_out = os_join(music_util.get_processed_path(), 'intermediate', dir_nm_, grp_nm)

            if 'many' in grp_nm:
                paths = sum([get_nested_paths(d) for d in [
                    # '000000-010000',
                    # '010000-020000',
                    # '020000-030000',
                    # '030000-040000',
                    # '040000-050000',
                    # '050000-060000',
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
                    grp_nm
                ]], start=[])
            else:
                paths = get_nested_paths(grp_nm)
            args['filenames'] = paths
        elif 'LMCI' in dnm:
            # grp_nm = 'many, lp'
            grp_nm = '020000-030000'
            # grp_nm = '110000-120000'
            # grp_nm = '120000-128478'

            dir_nm_ = f'23-04-18_LMCI_{{md={mode[0]}}}'
            path_out = os_join(music_util.get_processed_path(), 'intermediate', dir_nm_, grp_nm)

            if 'many' in grp_nm:
                paths = sum([get_nested_paths(d) for d in [
                    # '000000-010000',
                    # '010000-020000',
                    # '020000-030000',
                    # '030000-040000',
                    # '040000-050000',
                    # '050000-060000',
                    # '060000-070000',
                    # '070000-080000',
                    # '080000-090000',
                    # '090000-100000',
                    # '100000-110000',
                    # '110000-120000',
                    # '120000-128478'
                    grp_nm
                ]], start=[])
            else:
                paths = get_nested_paths(grp_nm)
            args['filenames'] = paths
        else:
            args['filenames'] = dnm
            if 'LMD-cleaned' in dnm:
                args['dataset_name2songs_args'] = dict(backend='all')

            resume = False
            # resume = True
            if resume:
                if dnm == 'POP909':
                    date = '23-03-31'
                elif dnm == 'MAESTRO':
                    date = '23-03-31'
                elif dnm == 'NES-MDB':
                    date = '23-04-17'
                else:
                    assert dnm == 'LMD-cleaned-subset'
                    date = '23-01-17'
                dir_nm_ = f'{date}_{dnm}_{{md={mode[0]}}}'
            else:
                date = now(fmt='short-date')
                dir_nm_ = f'{date}_{dnm}_{{md={mode[0]}}}'
            path_out = os_join(music_util.get_processed_path(), 'intermediate', dir_nm_)
        args['path_out'] = path_out
        me(**args)
    # export2json()

    def check_extract_progress():
        def is_folder_path(path_: str) -> bool:
            return os.path.isdir(path_) and not path_.endswith('.zip')

        def _folder2count(path_: str) -> int:
            # mic(path)
            # mic(os.listdir(path))
            _, _, fls = next(os.walk(path_))
            return len(fls)

        def _folder2file_counts(root_path: str) -> Dict[str, int]:
            return {
                d: _folder2count(os_join(root_path, d)) for d in sorted(os.listdir(root_path))
                if is_folder_path(os_join(root_path, d))
            }

        # check number of files that finished extraction
        path = os_join(music_util.get_processed_path(), 'intermediate')
        counts = dict()
        for fd_nm in sorted(os.listdir(path)):
            pa = os_join(path, fd_nm)
            if is_folder_path(pa):
                counts[fd_nm] = _folder2file_counts(pa) if 'LMD' in fd_nm else _folder2count(pa)
        print(pl.fmt(dict(counts=counts)))
    # check_extract_progress()

    def combine():
        def combine_single_json_songs(singe_song_dir: str, dataset_name: str):
            fl_pattern = '*.json'
            if any(k in dataset_name for k in ('LMD', 'LMCI')):
                fl_pattern = f'**/{fl_pattern}'
            path = os_join(music_util.get_processed_path(), 'intermediate', singe_song_dir, fl_pattern)
            fnms = sorted(glob.iglob(path))
            mic(len(fnms))
            # fnms = fnms[:1024]  # TODO: debugging
            fl = me.combine_saved_songs(filenames=fnms, output_filename=f'Extracted-{dataset_name}')
            mic(fl.keys(), len(fl['music']))
        # md = 'melody'
        md = 'full'
        if md == 'melody':
            combine_single_json_songs(singe_song_dir='22-10-02_POP909_{md=m}', dataset_name='POP909')
            combine_single_json_songs(singe_song_dir='22-10-02_MAESTRO_{md=m}', dataset_name='MAESTRO')
            # combine_single_json_songs(singe_song_dir='', dataset_name='LMD')
        else:
            # combine_single_json_songs(singe_song_dir='23-03-31_POP909_{md=f}', dataset_name='POP909')
            combine_single_json_songs(singe_song_dir='23-06-28_MAESTRO_{md=f}', dataset_name='MAESTRO')
            # combine_single_json_songs(singe_song_dir='23-04-06_LMD_{md=f}', dataset_name='LMD')
            # combine_single_json_songs(singe_song_dir='23-04-17_NES-MDB_{md=f}', dataset_name='NES')
            # combine_single_json_songs(singe_song_dir='23-04-18_LMCI_{md=f}', dataset_name='LMCI')
    # combine()

    def json2dset_with_split():
        """
        Split the data for now, when amount of data is not huge
        """
        seed = sconfig('random-seed')
        # mode = 'melody'
        mode = 'full'
        if mode == 'melody':
            # fnm = '22-10-03_Extracted-POP909_{n=909}_{md=m, prec=5, th=1}'
            fnm = '22-10-03_Extracted-MAESTRO_{n=1276}_{md=m, prec=5, th=1}'
            # fnm = ''
        else:
            # fnm = '22-10-22_Extracted-POP909_{n=909}_{md=f, prec=5, th=1}'
            fnm = '23-06-28_Extracted-MAESTRO_{n=1276}_{md=f, prec=5, th=1}'
            # fnm = '23-04-09_Extracted-LMD_{n=176640}_{md=f, prec=5, th=1}'

        predetermined_split = True
        if predetermined_split:
            import re
            pattern_dnm = re.compile(r'^\d{2}-\d{2}-\d{2}_Extracted-(?P<dnm>[^_]+)_(?P<md>[^_]+)_\{(.*)}$')
            m = pattern_dnm.match(fnm)
            assert m is not None
            dnm = m.group('dnm')
            mic(dnm)
            if dnm != 'MAESTRO':
                raise NotImplementedError('LMCI naming was obsolete, need to modify written json files')

            split_map = music_util.clean_dataset_paths('MAESTRO', return_split_map=True, verbose=False)

            dset = me.json2dataset(fnm, split=True, pre_determined_split=split_map)
        else:
            dset = me.json2dataset(fnm, split=True, test_size=0.02, test_size_range=(50, 500), seed=seed)
        mic(dset)
        mic(len(dset['train']), len(dset['test']))
        tr_samples, ts_samples = dset['train'][:2], dset['test'][:2]
        tr_samples['score'] = [s[:100] for s in tr_samples['score']]
        ts_samples['score'] = [s[:100] for s in ts_samples['score']]
        mic(tr_samples, ts_samples)
    json2dset_with_split()

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

        # dnm = 'LMD'
        dnm = 'LMCI'
        # dir_nm = '23-04-06_LMD_{md=f}'
        dir_nm = '23-04-18_LMCI_{md=f}'
        path_process_base = os_join(music_util.get_processed_path(), 'intermediate', dir_nm)
        path_to_process = os_join(path_process_base, 'many, lp')
        mic(path_to_process)
        paths = sorted(glob.iglob(os_join(path_to_process, '*.json'), recursive=True))
        if dnm == 'LMD':
            pattern = re.compile(r'^Music Export - (?P<ordinal>\d*)$')
        else:
            assert dnm == 'LMCI'
            pattern = re.compile(r'^Music Export - (?P<ordinal>\d*)_(?P<name>.*)$')
        o2f = music_util.Ordinal2Fnm(total=sconfig(f'datasets.{dnm}.meta.n_song'), group_size=int(1e4))
        it = tqdm(paths)
        for path in it:
            m = pattern.match(stem(path))
            assert m is not None
            o = int(m.group('ordinal'))

            fnm, dir_nm = o2f(o, return_parts=True)
            if dnm == 'LMCI':
                nm = m.group('name')
                fnm = f'{fnm}_{nm}'
            fnm = f'Music Export - {fnm}.json'
            it.set_postfix(fnm=pl.i(f'{dir_nm}/{fnm}'))
            path_out = os_join(path_process_base, dir_nm)
            os.makedirs(path_out, exist_ok=True)

            path_out = os_join(path_out, fnm)
            # mic(path, path_out)
            # if os.path.exists(path_out):
            #     continue
            assert not os.path.exists(path_out)
            shutil.move(path, path_out)
    # chore_move_proper_folder()

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
