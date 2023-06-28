import os
import re
import glob
import json
from os.path import join as os_join
from shutil import copyfile
from typing import Tuple, List, Dict, Union, Optional
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from stefutil import *
from musicnlp.util.util import *


logger = get_logger('Music File')


def get_processed_path():
    return os_join(get_base_path(), u.dset_dir, sconfig('datasets.my.dir_nm'))


def get_my_example_songs(k=None, pretty=False, fmt='mxl', extracted: bool = False, postfix: str = None):
    """
    :return: A list of or single MIDI file path
    """
    fmt = fmt.lower()
    ca(fmt=fmt)
    if extracted:
        assert fmt == 'mxl', 'Only support extracted for MXL files'
    dset_nm = f'{fmt}-eg'
    d_dset = sconfig(f'{u.dset_dir}.{dset_nm}')
    key_dir = 'dir_nm'
    if extracted:
        key_dir = f'{key_dir}_extracted'
    dir_nm = d_dset[key_dir]
    path = os_join(get_base_path(), u.dset_dir, dir_nm, d_dset[f'song_fmt_{fmt}'])
    paths = sorted(glob.iglob(path, recursive=True))
    if k is not None:
        assert isinstance(k, (int, str)), \
            f'Expect k to be either a {pl.i("int")} or {pl.i("str")}, got {pl.i(k)} with type {pl.i(type(k))}'
        if type(k) is int:
            return paths[k]
        else:  # Expect str
            k = k.lower()

            def match(p_: str) -> bool:
                p_ = stem(p_).lower()
                ret = p_.find(k) != -1
                if postfix:
                    ret = ret and p_.endswith(postfix)
                return ret
            return next(p for p in paths if match(p))
    else:
        return [stem(p) for p in paths] if pretty else paths


def get_extracted_song_eg(
        fnm='musicnlp music extraction, dnm=POP909, n=909, meta={mode=full, prec=5, th=1}, 2022-08-02_20-11-17',
        dir_=get_processed_path(),
        k: Union[int, str] = 0
) -> str:
    with open(os_join(dir_, f'{fnm}.json')) as f:
        dset = json.load(f)['music']
    if isinstance(k, int):
        return dset[k]['score']
    else:
        return next(d['score'] for d in dset if k in d['title'])


def lmd_cleaned_title2title_n_ver(title: str) -> Tuple[str, int]:
    """
    :param title: File name of format `<title>(.<ver>)?`

    Note there's a file in the original dataset, by artist `Gary Glitter`, named `Rock 'n' Roll PT.2..mid`
        We consider this an edge case, the last dot should not be there, and we ignore this case
        Users are advised to modify the file to remove the dot
    """
    if not hasattr(lmd_cleaned_title2title_n_ver, 'pattern_title'):  # version as a number has to be at the end
        lmd_cleaned_title2title_n_ver.pattern_title = re.compile(r'^(?P<title>.*)\.(?P<version>[1-9]\d*)$')
    m = lmd_cleaned_title2title_n_ver.pattern_title.match(title)
    if m:
        title_, version = m.group('title'), m.group('version')
        assert title_ is not None and version is not None
        title, v = title_, int(version)
    else:
        v = 0
    return title, v


class Ordinal2Fnm:
    """
    Convert an ordinal to a filesystem name

    A nested directory to save file system rendering

    Intended for LMD file cleaning
        storing 170k songs in a single folder makes prohibitively slow FS rendering
    """
    def __init__(self, total: int, group_size: int = 1e4, ext: str = None):
        self.total, self.grp_sz = total, group_size
        self.n_digit = len(str(total))

        self.ext = ext

    def __call__(self, i: int, return_parts: bool = False) -> Union[str, Tuple[str, str]]:
        i_grp = i // self.grp_sz
        strt, end = i_grp * self.grp_sz, min((i_grp + 1) * self.grp_sz, self.total)
        dir_nm = f'{self._num2padded(strt)}-{self._num2padded(end)}'
        fnm = f'{i:>0{self.n_digit}}'
        if self.ext:
            fnm = f'{fnm}.{self.ext}'
        return (fnm, dir_nm) if return_parts else os_join(dir_nm, fnm)

    def _num2padded(self, n_: int):
        return f'{n_:0{self.n_digit}}'


def clean_dataset_paths(
        dataset_name: str = 'POP909', return_split_map: bool = False, verbose: bool = True
) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Convert datasets in their original sources to my own file system hierarchy & names
        A directory of `midi` files, with title and artist as file name

    :return: If `return_fnm_map`, instead of copying files to new location,
        return a mapping from new filename, to original filename and corresponding dataset split

        Intended for datasets with music split already assigned
    """
    ca.check_mismatch('Dataset Name', dataset_name, ['POP909', 'LMD-cleaned', 'LMD', 'MAESTRO', 'LMCI', 'NES-MDB'])

    path_exp = os_join(get_base_path(), u.dset_dir, 'converted', dataset_name)
    if return_split_map:
        ret = dict()
    else:
        os.makedirs(path_exp, exist_ok=True)
        ret = None

    if dataset_name == 'POP909':
        assert not return_split_map
        path = os_join(get_base_path(), u.dset_dir, 'POP909-Dataset', dataset_name)
        df = pd.read_excel(os_join(path, 'index.xlsx'))
        paths = sorted(glob.iglob(os_join(path, '*/*.mid'), recursive=True))
        for i, p in enumerate(tqdm(paths)):
            rec = df.iloc[i, :]
            fnm = f'{rec["artist"]} - {rec["name"]}.mid'
            copyfile(p, os_join(path_exp, fnm))
    elif dataset_name == 'LMD-cleaned':
        assert not return_split_map
        d_dset = sconfig(f'datasets.{dataset_name}.original')
        path_ori = os_join(get_base_path(), u.dset_dir, d_dset['dir_nm'])
        paths = sorted(glob.iglob(os_join(path_ori, d_dset['song_fmt_mid'])))

        # empirically seen as a problem: some files are essentially the same title, ending in different numbers
        # See `ValueError` below
        my_lim, os_lim = 256-32, 255

        def path2fnm(p_: str):
            if not hasattr(path2fnm, 'count_too_long'):
                path2fnm.count_too_long = 0
            paths_last = p_.split(os.sep)[-2:]
            artist, title_ = paths_last
            title_ = title_[:-4]  # remove `.mid`
            title_, v = lmd_cleaned_title2title_n_ver(title_)

            fnm__ = clean_whitespace(f'{artist} - {title_}')
            assert len(clean_whitespace(artist)) - 3 <= my_lim, \
                f'Artist name {pl.i(artist)} is too long for OS file write'
            if len(fnm__) > my_lim:
                # Modified the name, but still keep to the original way of versioning,
                #   i.e. `<title>.<version>` if there's a separate version,
                # so that `get_lmd_cleaned_subset_fnms` can work without changes
                # TODO: however, the original LMD dataset's way of versioning the same song
                #  is not intuitive & better be changed
                fnm__ = f'{fnm__[:my_lim]}... '
                path2fnm.count_too_long += 1
            v_str = '' if v == 0 else f'.{v}'
            fnm__ = f'{fnm__}{v_str}'
            fnm__ = f'{fnm__}.mid'
            assert len(fnm__) <= os_lim
            return fnm__
        fnms_written = set()
        for p in tqdm(paths, desc=f'Converting {dataset_name}', unit='song'):
            fnm = path2fnm(p)
            if fnm in fnms_written:
                raise ValueError(f'Duplicate file name because of truncation: path {pl.i(p)} modified to {pl.i(fnm)}')
            fnms_written.add(fnm)
            copyfile(p, os_join(path_exp, fnm))
        assert len(fnms_written) == len(paths)
        print(f'{pl.i(path2fnm.count_too_long)} files were truncated to {pl.i(my_lim)} characters')
    elif dataset_name == 'LMD':
        assert not return_split_map
        d_dset = sconfig(f'datasets.{dataset_name}.original')
        path_ori = os_join(get_base_path(), u.dset_dir, d_dset['dir_nm'])
        paths = sorted(glob.iglob(os_join(path_ori, d_dset['song_fmt_mid']), recursive=True))
        o2f = Ordinal2Fnm(total=len(paths), group_size=int(1e4))

        it = tqdm(paths)
        for i, p in enumerate(it):
            fnm, dir_nm = o2f(i, return_parts=True)
            it.set_postfix(fnm=f'{dir_nm}/{fnm}')
            path = os_join(path_exp, dir_nm)
            os.makedirs(path, exist_ok=True)
            copyfile(p, os_join(path, fnm))
    elif dataset_name in ['MAESTRO', 'NES-MDB']:
        d_dset = sconfig(f'datasets.{dataset_name}')
        n_song = get(d_dset, 'meta.n_song')
        d_dset = d_dset['original']
        path_ori = os_join(get_base_path(), u.dset_dir, d_dset['dir_nm'])
        paths = sorted(glob.iglob(os_join(path_ori, d_dset['song_fmt_mid']), recursive=True))
        assert len(paths) == n_song  # sanity check
        if verbose:
            logger.info(f'Found {pl.i(n_song)} unique MIDI files')

        set_fnm, set_dup_fnm = set(), set()
        d_fnm = dict()

        is_maestro = dataset_name == 'MAESTRO'
        tqdm_args = dict(total=n_song, desc='Checking for duplicate file names', unit='song')
        df = None
        if is_maestro:
            df = pd.read_csv(os_join(path_ori, 'maestro-v3.0.0.csv'))
            assert len(df) == n_song

            def _row2fnm(r: pd.Series) -> str:
                composer = r.canonical_composer.replace('/', '&')
                title = r.canonical_title.replace('/', ':')
                # wicked case in the dataset: title all the same apart from capitalization
                title = ' '.join([w.capitalize() for w in title.split()])
                return f'{composer} - {title}'
            for i, row in tqdm(df.iterrows(), **tqdm_args):
                fnm = _row2fnm(row)  # consider as same piece if same composer & title
                if fnm in set_fnm:
                    set_dup_fnm.add(fnm)
                else:
                    set_fnm.add(fnm)
                d_fnm[i] = fnm
        else:  # NES-MDB
            # extract the two-level name from the file name
            # e.g. 005_Abadox_TheDeadlyInnerWar_00_01OpeningSE
            pattern = re.compile(r'^\d{3}_(?P<title>.*)_(?P<sec_start>\d{2})_(?P<sec_end>\d{2})(?P<suffix>.*)$')
            # e.g. 211_M82GameSelectableWorkingProductDisplay_00_M82GameSelectableWorkingProductDisplay01MainMusic
            pattern_fall = re.compile(r'^\d{3}_(?P<title>.*)_(?P<sec>\d{2})(?P<suffix>.*)$')

            def fnm2fnm(f: str) -> str:
                m, fall = pattern.match(f), False
                if m is None:
                    m, fall = pattern_fall.match(f), True
                assert m is not None
                ttl, suffix = m.group('title'), m.group('suffix')

                # section number needed, otherwise 24 files have duplicate names
                if not fall:
                    # not necessarily the case,
                    # e.g. `train/055_CircusCaper_01_01bTitleScreen.mid`, `train/055_CircusCaper_12_07bStage4.mid`
                    # st, ed = int(m.group('sec_start')), int(m.group('sec_end'))
                    # assert st + 1 == ed
                    st, ed = m.group('sec_start'), m.group('sec_end')
                    n = f'{st}-{ed}'
                else:
                    n = m.group('sec')
                return f'{ttl}-{n}-{suffix}'
            for p in tqdm(paths, **tqdm_args):
                fnm = fnm2fnm(stem(p))
                if fnm in set_fnm:
                    set_dup_fnm.add(fnm)
                else:
                    set_fnm.add(fnm)
                d_fnm[p] = fnm
        if verbose:
            logger.info(f'Found {pl.i(len(set_dup_fnm))} duplicate file names')

        dup_fnm2ver = defaultdict(int)
        desc = f'Extracting filename map' if return_split_map else f'Copying w/ new name'

        it = df.iterrows() if is_maestro else paths
        it = tqdm(it, total=n_song, desc=desc, unit='song')
        split_map = dict(train='train', test='test', valid='validation')  # for NES-MDB
        for idx_, i in enumerate(it):
            row = None
            if is_maestro:
                i, row = i
                fnm_ori = row.midi_filename
            else:
                i: str
                fnm_ori = stem(i)
            it.set_postfix(fnm=pl.i(fnm_ori))
            fnm = d_fnm[i]

            if fnm in set_dup_fnm:
                fnm_ = f'{fnm}_v{dup_fnm2ver[fnm]}'
                dup_fnm2ver[fnm] += 1
                if verbose:
                    logger.info(f'Piece {pl.i(fnm)} is duplicated, renaming to {pl.i(fnm_)}')
            else:
                fnm_ = fnm
            if return_split_map:
                if is_maestro:
                    split = row.split
                else:
                    dirs = i.split(os.sep)
                    split = split_map[dirs[-2]]
                ret[fnm_] = dict(original_fnm=fnm_ori, split=split)
            else:
                pa_ori = os_join(path_ori, fnm_ori) if is_maestro else i
                copyfile(pa_ori, os_join(path_exp, f'{fnm_}.mid'))
                assert os.path.exists(os_join(path_exp, f'{fnm_}.mid'))
                mic(idx_, fnm_)
        if return_split_map:  # sanity check no filename duplication
            # sanity check new filename no duplication; note that original filenames are definitely unique
            assert len(ret) == n_song
        else:
            assert len(list(i for i in glob.iglob(os_join(path_exp, '*.mid')))) == n_song
    else:
        assert dataset_name == 'LMCI'
        assert not return_split_map
        d_dset = sconfig(f'datasets.{dataset_name}.original')
        path_ori = os_join(get_base_path(), u.dset_dir, d_dset['dir_nm'])

        exts = ('.mid', '.midi')  # some are in upper case
        paths = sorted(p for p in glob.iglob(os_join(path_ori, '**/*'), recursive=True) if p.lower().endswith(exts))
        n_uniq_midi = len(paths)
        if verbose:
            logger.info(f'Found {pl.i(n_uniq_midi)} unique MIDI files')

        set_fnm, set_dup_fnm = set(), set()
        for p in tqdm(paths, desc='Checking for duplicate file names'):
            fnm = stem(p)  # with extension
            if fnm in set_fnm:
                set_dup_fnm.add(fnm)
            else:
                set_fnm.add(fnm)
        dup_fnm2ver = defaultdict(int)
        if verbose:
            logger.info(f'Found {pl.i(len(set_dup_fnm))} duplicate file names')

        o2f = Ordinal2Fnm(total=len(paths), group_size=int(1e4))
        it = tqdm(paths, desc='Copying with new name')
        for i, p in enumerate(it):
            fnm = stem(p)

            if i == 88685:  # file name too long, reduce
                assert fnm == 'Nausicaa Of The Valley Of The Wind - &#1044;&#1072;&#1074;&#1085;&#1086; &#1091;' \
                               '&#1096;&#1077;&#1076;&#1096;&#1080;&#1077; &#1076;&#1085;&#1080; (&#1054;&#1089;' \
                               '&#1085;&#1086;&#1074;&#1085;&#1072;&#1103; &#1090;&#1077;&#1084;&#1072;)'
                assert fnm not in set_dup_fnm
                fnm = 'Nausicaa Of The Valley Of The Wind'  # manually shorten for now
            i: int
            pref, dir_nm = o2f(i, return_parts=True)
            if fnm in set_dup_fnm:
                fnm_ = f'{fnm}_v{dup_fnm2ver[fnm]}'
                dup_fnm2ver[fnm] += 1
                if verbose:
                    logger.info(f'Filename {pl.i(fnm)} is duplicated, renaming to {pl.i(fnm_)}')
            else:
                fnm_ = fnm
            fnm = f'{pref}_{fnm_}.mid'
            it.set_postfix(fnm=f'{dir_nm}/{fnm}')

            path = os_join(path_exp, dir_nm)
            os.makedirs(path, exist_ok=True)
            copyfile(p, os_join(path, fnm))
        # sanity check no name collision
        assert n_uniq_midi == len(list(i for i in glob.iglob(os_join(path_exp, '**/*.mid'))))
    return ret


def get_lmd_cleaned_subset_fnms() -> List[str]:
    """
    My subset of LMD-cleaned dataset
        MIDI files are converted to MXL via MuseScore, and fallback Logic Pro
        Only one unique artist-song is picked among the many versions
            Resolve by just taking the first one

    Filenames respect those produced from `clean_dataset_paths`, expect it to be called first
    """
    # this folder contains all MIDI files that can be converted to MXL, on my machine
    # <artist> - <title>(.<version>)?.mid
    pattern = re.compile(r'^(?P<artist>.*) - (?P<title>.*)(\.(?P<version>[1-9]\d*))?\.mid$')
    d_song2fnms: Dict[Tuple[str, str], Dict[int, str]] = defaultdict(dict)

    path_ms = os_join(get_base_path(), u.dset_dir, 'converted', 'LMD-cleaned, MS')
    path_lp = os_join(get_base_path(), u.dset_dir, 'converted', 'LMD-cleaned, LP')
    it_ms, it_lp = glob.iglob(os_join(path_ms, '*.mid')), glob.iglob(os_join(path_lp, '*.mid'))
    fnms = sorted(chain_its([it_ms, it_lp]))
    for fnm in tqdm(fnms, desc='Getting LMD-cleaned subset', unit='song'):
        fnm = stem(fnm, keep_ext=True)
        m = pattern.match(fnm)
        artist, title = m.group('artist'), m.group('title')
        title, version = lmd_cleaned_title2title_n_ver(title)

        version = version or 0
        d = d_song2fnms[(artist, title)]
        assert version not in d
        d[version] = fnm
    return [d[min(d)] for d in d_song2fnms.values()]


def get_converted_song_paths(
        dataset_name: str = None, fmt='mxl', backend: Optional[str] = 'MS', _inner: bool = False
) -> List[str]:
    """
    :param dataset_name: dataset name
    :param fmt: song file format, one of ['mid', 'mxl']
        `mxl`s are converted from MIDI files
    :param backend: midi to MXL conversion backend, one of ['MS', 'LP', 'all'] for MuseScore or Logic Pro
        This determines the stored path
        `all` is relevant for LMD only, see `get_lmd_conversion_meta`
        If `all`, the conversion from both backends are combined
    :param _inner: recursion flag for internal use only
    :return: List of music file paths in my converted file system structure, see `clean_dataset_paths`

    xml converted from MIDI files
        Default conversion with MuseScore, fallback to Logic Pro
    """
    ca(dataset_name=dataset_name, fmt=fmt)
    ca.check_mismatch('Conversion Backend', backend, ['MS', 'LP', 'all'])
    d_log = dict(dataset_name=dataset_name, fmt=fmt, backend=backend)
    if not _inner:
        logger.info(f'Getting converted song paths w/ {pl.i(d_log)}... ')

    d_dset = sconfig(f'datasets.{dataset_name}.converted')
    dir_nm = d_dset['dir_nm']
    if backend == 'all':
        fls_ms = get_converted_song_paths(dataset_name, fmt=fmt, backend='MS', _inner=True)
        fls_lp = get_converted_song_paths(dataset_name, fmt=fmt, backend='LP', _inner=True)
        return sorted(fls_ms + fls_lp, key=lambda f: stem(f))
    else:
        dset_path = os_join(get_base_path(), u.dset_dir)
        if backend is not None:
            dir_nm = f'{dir_nm}, {backend}'
        path = os_join(dset_path, dir_nm, d_dset[f'song_fmt_{fmt}'])
        return sorted(glob.iglob(path, recursive=True))


def get_conversion_meta(dataset_name: str = 'LMD'):
    """
    Converting POP909 and MAESTRO all terminated without error in MuseScore

    But many broken files in LMD that can't be read by MuseScore, those files fall back to Logic Pro conversion
        We don't start with Logic Pro conversion for lack of efficient batch processing support
    Still, a subset of files can't be converted with Logic Pro
    """
    ca.check_mismatch('Dataset Name', dataset_name, ['LMD', 'LMCI'])

    n_song = sconfig(f'datasets.{dataset_name}.meta.n_song')
    dir_nm = sconfig(f'datasets.{dataset_name}.converted.dir_nm')
    conv_paths = get_converted_song_paths(dataset_name=dataset_name, fmt='mxl', backend='all')
    set_converted = set(conv_paths)
    lst_meta = []
    o2f = Ordinal2Fnm(total=n_song, group_size=int(1e4), ext='mxl')

    def get_original_fnms():  # see `clean_dataset_paths`
        d_dset = sconfig(f'datasets.{dataset_name}.original')
        path_ori = os_join(get_base_path(), u.dset_dir, d_dset['dir_nm'])

        if dataset_name == 'LMCI':
            exts = ('.mid', '.midi')
            # can't iterate original file names cos I made too much changes, e.g. adding version postfix
            return sorted(p for p in glob.iglob(os_join(path_ori, '**/*'), recursive=True) if p.lower().endswith(exts))
        else:  # `LMD`
            return sorted(glob.iglob(os_join(path_ori, d_dset['song_fmt_mid']), recursive=True))
    ori_dir_nm = sconfig(f'datasets.{dataset_name}.original.dir_nm')

    def original_abs2rel(path: str) -> str:
        return path[path.index(ori_dir_nm)+len(ori_dir_nm)+1:]
    original_fnms = get_original_fnms()
    assert len(original_fnms) == n_song  # sanity check
    it = tqdm(enumerate(original_fnms), desc='Scanning converted files', unit='fl', total=n_song)

    map_paths = None
    if dataset_name == 'LMCI':
        brk_base = os_join(get_base_path(), u.dset_dir, f'{dir_nm}, broken')
        brk_paths = list(glob.iglob(os_join(brk_base, '**/*.mid'), recursive=True))
        if len(brk_paths) + len(conv_paths) != n_song:
            raise ValueError(f'Number of converted files {pl.i(len(conv_paths))} + broken files {pl.i(len(brk_paths))} '
                             f'!= total number of songs {pl.i(n_song)}')

        ds = dir_nm.split(os.sep)
        assert len(ds) == 2  # sanity check
        d1, d2 = ds

        def full2rel(path: str) -> str:
            dirs = path.split(os.sep)
            for i_, d in enumerate(dirs):
                if d == d1 and dirs[i_+1].startswith(d2):
                    return os_join(*dirs[i_+2:])  # e.g. `'000000-010000/000000_009count.mxl'`
            raise ValueError(f'Path {pl.i(path)} does not contain {pl.i(dir_nm)}')
        map_paths = sorted(conv_paths + brk_paths, key=full2rel)
        for i, mp in enumerate(map_paths):
            assert i == int(stem(mp)[:6])  # sanity check
    for i, ori_fnm in it:  # ensure go through every file
        if dataset_name == 'LMCI':
            _fnm, _dir_nm = o2f(i, return_parts=True)
            mf = stem(map_paths[i])
            assert _fnm[:6] == mf[:6]  # sanity check sorting results in one-to-one mapping
            # for edge cases: `Biogra11.mid.mid`, `brighton.midi.mid`
            of = stem(ori_fnm).removesuffix('.mid').removesuffix('.midi')
            # sanity check only thing added should be version postfix: drop ordinal, up until original file name
            assert i == 88685 or mf[7:][:len(of)] == of  # Discard edge case, see `clean_dataset_paths`
            fnm = os_join(_dir_nm, f'{mf}.mxl')
        else:
            fnm = o2f(i)
        it.set_postfix(fnm=pl.i(stem(fnm)))
        # default conversion store location
        path_ms, path_lp = os_join(u.dset_dir, f'{dir_nm}, MS', fnm), os_join(u.dset_path, f'{dir_nm}, LP', fnm)
        _path_ms, _path_lp = os_join(u.base_path, path_ms), os_join(u.base_path, path_lp)
        d_out = dict(file_name=fnm, original_filename=original_abs2rel(ori_fnm))
        if os.path.exists(_path_ms):
            d_out.update(dict(backend='MS', path=path_ms, status='converted'))
            set_converted.remove(_path_ms)
        elif os.path.exists(_path_lp):
            d_out.update(dict(backend='LP', path=path_lp, status='converted'))
            set_converted.remove(_path_lp)
        else:
            # the original `mid` file should still be there to mark error
            path_broken = _path_lp.replace(f'{dataset_name}, LP', f'{dataset_name}, broken').replace('.mxl', '.mid')
            if not os.path.exists(path_broken):  # TODO: debugging
                raise FileNotFoundError(f'Broken file not found: {pl.i(path_broken)}')
            assert os.path.exists(path_broken)
            # note all drums is also considered empty
            d_out.update(dict(backend='NA', path='NA', status='error/empty'))
        lst_meta.append(d_out)
    assert len(set_converted) == 0  # sanity check no converted file is missed
    df = pd.DataFrame(lst_meta)
    date = now(fmt='short-date')
    path_out = os_join(u.dset_path, 'converted', f'{date}, {dataset_name} conversion meta.csv')
    df.to_csv(path_out)
    return df


if __name__ == '__main__':
    mic.output_width = 256

    def check_processed_path():
        mic(get_processed_path())
    # check_processed_path()

    def check_fl_nms():
        # dnm = 'POP909'
        dnm = 'MAESTRO'
        fnms = get_converted_song_paths(dnm, fmt='mid')
        mic(len(fnms), fnms[:20])
        fnms = get_converted_song_paths(dnm, fmt='mxl')
        mic(len(fnms), fnms[:20])
    # check_fl_nms()

    def clean_paths():
        # clean_dataset_paths('LMD-cleaned')
        # clean_dataset_paths('LMD')
        clean_dataset_paths('MAESTRO')
        # clean_dataset_paths('LMCI')
        # clean_dataset_paths('NES-MDB')
    clean_paths()

    def clean_paths2dset_split():
        d = clean_dataset_paths('MAESTRO', return_split_map=True)
        # d = clean_dataset_paths('NES-MDB', return_fnm_map=True)
        mic(d)
    # clean_paths2dset_split()

    def get_lmd_subset():
        fnms = get_lmd_cleaned_subset_fnms()
        # fnms = get_converted_song_paths('LMD-cleaned-subset')
        mic(len(fnms), fnms[:20])
    # get_lmd_subset()

    def lmd_cleaned_subset_to_files():
        """
        Construct the subset & move them to separate folders for easier later processing
        """
        # fd_ori = 'LMD-cleaned, MS'
        # fd_sub = 'LMD-cleaned-subset, MS'
        fd_ori = 'LMD-cleaned, LP'
        fd_sub = 'LMD-cleaned-subset, LP'
        path_base = os_join(u.dset_path, 'converted')
        os.makedirs(os_join(path_base, fd_sub), exist_ok=True)
        fnms_keep = get_lmd_cleaned_subset_fnms()
        mic(len(fnms_keep))

        fnms = sorted(glob.iglob(os_join(path_base, fd_ori, '*.mid')))
        for f in tqdm(fnms, desc='Copying subset songs'):
            # mic(f)
            # mic(stem(f, keep_ext=True))
            if stem(f, keep_ext=True) in fnms_keep:
                stm = stem(f)
                for f_ in [f'{stm}.mid', f'{stm}.mxl', f'{stm}.xml']:
                    f__ = os_join(path_base, fd_ori, f_)
                    f_out = os_join(path_base, fd_sub, f_)
                    if os.path.exists(f__):
                        copyfile(f__, f_out)
    # lmd_cleaned_subset_to_files()

    def mv_backend_not_processed():
        """
        Some files are not processed properly, e.g. missing xml, incorrect file name
        Pick out those and move to a different folder to process again

        See `batch-processing`

        Files to process are in `todo`, move the processed ones back to default

        After batch-convert terminates, check for the files processed in last session
        """
        import shutil
        # dnm = 'LMD-cleaned_broken'
        # dnm = 'POP909, LP'
        # dnm = 'MAESTRO'
        # dnm = 'LMD, MS/040000-050000'
        # dnm = 'LMD, LP/170000-178561'
        # dnm = 'LMD-cleaned, LP'
        # dnm = 'LMCI, MS/110000-120000'
        # dnm = 'LMCI, MS/120000-128478'
        dnm = 'LMCI, LP/110000-120000'
        path_processed = os_join(u.dset_path, 'converted', dnm)
        """
        Among group of 10k files in a folder for conversion, MS in Mac produces ~100 broken songs, 
        but MS in Win consistently produces ~150 broken songs, pass them through Mac again, 
        and some of the files can be converted now...  
        """
        # dnm = 'LMD, broken, Win/060000-070000'
        # path_processed = os_join(u.dset_path, dnm)
        path_to_process = f'{path_processed}, todo'
        mic(path_processed)
        os.makedirs(path_processed, exist_ok=True)
        path_mids = sorted(glob.iglob(os_join(path_to_process, '**.mid')))
        logger.info(f'{pl.i(len(path_mids))} MIDI files should have been converted')
        count = 0
        output_format = 'xml'
        # output_format = 'mxl'
        fnm = None
        for path in tqdm(path_mids):
            path_xml = path[:-len('.mid')]
            path_xml = f'{path_xml}.{output_format}'
            # path_xml = path.replace('.mid', f'.{output_format}')
            if os.path.exists(path_xml):
                fnm = stem(path)
                # logger.info(f'{pl.i(fnm)} converted, moved to processed folder')
                shutil.move(path, os_join(path_processed, f'{fnm}.mid'))  # move to processed folder
                shutil.move(path_xml, os_join(path_processed, f'{fnm}.{output_format}'))
                count += 1
        logger.info(f'{pl.i(count)} MIDIs converted in the last session')
        count = 0
        path_xmls = sorted(glob.iglob(os_join(path_to_process, f'*.{output_format}')))
        for path in tqdm(path_xmls):
            path_mid = path.replace(f'.{output_format}', '.mid')
            if not os.path.exists(path_mid):
                os.remove(path)
                logger.info(f'Original MIDI for {pl.i(fnm)} not found, removed')
                count += 1
        logger.info(f'{pl.i(count)} converted xml with unknown origin in the last session removed')
    # mv_backend_not_processed()

    def get_convert_meta():
        # dnm = 'LMD'
        dnm = 'LMCI'
        df = get_conversion_meta(dataset_name=dnm)
        mic(df)
    # get_convert_meta()
