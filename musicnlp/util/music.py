import os
import re
import glob
import json
from shutil import copyfile
from typing import Tuple, List, Dict, Union
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from musicnlp.util.util import *
from musicnlp.util.data_path import PATH_BASE, DIR_DSET


def get_processed_path():
    return os.path.join(PATH_BASE, DIR_DSET, config('datasets.my.dir_nm'))


def get_my_example_songs(k=None, pretty=False, fmt='mxl', extracted: bool = False):
    """
    :return: A list of or single MIDI file path
    """
    fmt = fmt.lower()
    ca(fmt=fmt)
    if extracted:
        assert fmt == 'mxl', 'Only support extracted for MXL files'
    dset_nm = f'{fmt}-eg'
    d_dset = config(f'{DIR_DSET}.{dset_nm}')
    key_dir = 'dir_nm'
    if extracted:
        key_dir = f'{key_dir}_extracted'
    dir_nm = d_dset[key_dir]
    path = os.path.join(PATH_BASE, DIR_DSET, dir_nm, d_dset[f'song_fmt_{fmt}'])
    paths = sorted(glob.iglob(path, recursive=True))
    if k is not None:
        assert isinstance(k, (int, str)), \
            f'Expect k to be either a {logi("int")} or {logi("str")}, got {logi(k)} with type {logi(type(k))}'
        if type(k) is int:
            return paths[k]
        else:  # Expect str
            k = k.lower()
            return next(p for p in paths if p.lower().find(k) != -1)
    else:
        return [stem(p) for p in paths] if pretty else paths


def get_extracted_song_eg(
        fnm='musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-51-01',
        dir_=get_processed_path(),
        k: Union[int, str] = 0
) -> str:
    with open(os.path.join(dir_, f'{fnm}.json')) as f:
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


def convert_dataset(dataset_name: str = 'POP909'):
    """
    Convert datasets in their original sources to my own file system hierarchy & names
        A directory of `midi` files, with title and artist as file name
    """
    dnms = ['POP909', 'LMD-cleaned']
    assert dataset_name in dnms, f'Unsupported dataset name: expect one of {logi(dnms)}, got {logi(dataset_name)}'

    path_exp = os.path.join(PATH_BASE, DIR_DSET, dataset_name)
    os.makedirs(path_exp, exist_ok=True)
    if dataset_name == 'POP909':
        path = os.path.join(PATH_BASE, DIR_DSET, 'POP909-Dataset', dataset_name)
        df = pd.read_excel(os.path.join(path, 'index.xlsx'))
        paths = sorted(glob.iglob(os.path.join(path, '*/*.mid'), recursive=True))
        for i, p in enumerate(tqdm(paths)):
            rec = df.iloc[i, :]
            fnm = f'{rec["artist"]} - {rec["name"]}.mid'
            copyfile(p, os.path.join(path_exp, fnm))
    elif dataset_name == 'LMD-cleaned':
        d_dset = config(f'datasets.{dataset_name}')
        path_ori = os.path.join(PATH_BASE, DIR_DSET, d_dset['dir_nm'])
        fnms = sorted(glob.iglob(os.path.join(path_ori, d_dset['song_fmt_mid'])))

        # empirically seen as a problem: some files are essentially the same title, ending in different numbers
        # See `ValueError` below
        my_lim, os_lim = 256-32, 255

        def path2fnm(p_: str):
            if not hasattr(path2fnm, 'count_too_long'):
                path2fnm.count_too_long = 0
            paths_last = p_.split(os.sep)[-2:]
            artist, title = paths_last
            title = title[:-4]  # remove `.mid`
            title, v = lmd_cleaned_title2title_n_ver(title)

            fnm_ = clean_whitespace(f'{artist} - {title}')
            assert len(clean_whitespace(artist)) - 3 <= my_lim, \
                f'Artist name {logi(artist)} is too long for OS file write'
            if len(fnm_) > my_lim:
                # Modified the name, but still keep to the original way of versioning,
                #   i.e. `<title>.<version>` if there's a separate version,
                # so that `get_lmd_cleaned_subset_fnms` can work without changes
                # TODO: however, the original LMD dataset's way of versioning the same song
                #  is not intuitive & better be changed
                fnm_ = f'{fnm_[:my_lim]}... '
                path2fnm.count_too_long += 1
            v_str = '' if v == 0 else f'.{v}'
            fnm_ = f'{fnm_}{v_str}'
            fnm_ = f'{fnm_}.mid'
            assert len(fnm_) <= os_lim
            return fnm_
        fnms_written = set()
        for p in tqdm(fnms, desc=f'Converting {dataset_name}', unit='song'):
            fnm = path2fnm(p)
            if fnm in fnms_written:
                raise ValueError(f'Duplicate file name because of truncation: path {logi(p)} modified to {logi(fnm)}')
            fnms_written.add(fnm)
            copyfile(p, os.path.join(path_exp, fnm))
        assert len(fnms_written) == len(fnms)
        print(f'{logi(path2fnm.count_too_long)} files were truncated to {logi(os_lim)} characters')


def get_lmd_cleaned_subset_fnms() -> List[str]:
    """
    My subset of LMD-cleaned dataset
        MIDI files that can't be converted to MXL via MuseScore are excluded
        Only one unique artist-song is picked among the many versions
            Resolve by just taking the first one

    Expects `convert_dataset` called first
    """
    # this folder contains all MIDI files that can be converted to MXL, on my machine
    path = os.path.join(PATH_BASE, DIR_DSET, 'LMD-cleaned_valid')
    # <artist> - <title>(.<version>)?.mid
    pattern = re.compile(r'^(?P<artist>.*) - (?P<title>.*)(\.(?P<version>[1-9]\d*))?\.mid$')
    d_song2fnms: Dict[Tuple[str, str], Dict[int, str]] = defaultdict(dict)
    fnms = sorted(glob.iglob(os.path.join(path, '*.mid')))
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


def get_cleaned_song_paths(dataset_name: str, fmt='mid') -> List[str]:
    """
    :return: List of music file paths in my cleaned file system structure
    """
    lmd_c_s = 'LMD-cleaned-subset'
    dataset_names = list(config('datasets').keys()) + [lmd_c_s]
    assert dataset_name in dataset_names, \
        f'Invalid dataset name: {logi(dataset_name)}, expected one of {logi(dataset_names)}'
    fmts = ['mid', 'mxl']
    assert fmt in fmts, f'Invalid format: {logi(fmt)}, expected one of {logi(fmts)}'

    path = os.path.join(PATH_BASE, DIR_DSET)

    if dataset_name == lmd_c_s:
        fnms = get_lmd_cleaned_subset_fnms()
        dir_nm = 'LMD-cleaned_valid'

        if fmt == 'mid':
            def map_fnm(fnm: str) -> str:
                return os.path.join(path, dir_nm, fnm)
        else:  # 'mxl'
            def map_fnm(fnm: str) -> str:
                return os.path.join(path, dir_nm, f'{stem(fnm)}.{fmt}')
        return [map_fnm(fnm) for fnm in fnms]
    else:
        d_dset = config(f'datasets.{dataset_name}')
        dir_nm = d_dset['dir_nm']
        path = os.path.join(path, dir_nm, d_dset[f'song_fmt_{fmt}'])
        # from icecream import ic
        # ic(path)
        # ic(os.listdir(os.path.join(PATH_BASE, DIR_DSET, dir_nm)))
        # ic(len(sorted(glob.iglob(path, recursive=True))))
        # exit(1)
        return sorted(glob.iglob(path, recursive=True))


if __name__ == '__main__':
    from icecream import ic

    ic.lineWrapWidth = 150

    def check_fl_nms():
        dnm = 'POP909'
        fnms = get_cleaned_song_paths(dnm)
        ic(len(fnms), fnms[:20])
        fnms = get_cleaned_song_paths(dnm, fmt='song_fmt_exp')
        ic(len(fnms), fnms[:20])
    # check_fl_nms()

    # convert_dataset('LMD-cleaned')

    # import music21 as m21
    # path_broken = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/broken/LMD-cleaned/broken'
    # # broken_fl = 'ABBA - I\'ve Been Waiting For You.mid'
    # # broken_fl = 'Aerosmith - Pink.3.mid'
    # broken_fl = 'Alice in Chains - Sludge Factory.mid'
    # # broken_fl = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/broken/LMD-cleaned/fixed/' \
    # #             'ABBA - I\'ve Been Waiting For You.band.mid'
    # ic(broken_fl)
    # scr = m21.converter.parse(os.path.join(path_broken, broken_fl))
    # ic(scr)

    def fix_delete_broken_files():
        import glob

        path_broken = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/LMD-cleaned_broken/*.mid'
        set_broken = set(clean_whitespace(stem(fnm)) for fnm in glob.iglob(path_broken))
        ic(set_broken)
        path_lmd_c = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/LMD-cleaned/*.mid'
        for fnm in glob.iglob(path_lmd_c):
            if stem(fnm) in set_broken:
                os.remove(fnm)
                set_broken.remove(stem(fnm))
                print('Deleted', fnm)
        ic(set_broken)
        assert len(set_broken) == 0, 'Not all broken files deleted'
    # fix_delete_broken_files()

    def fix_match_mxl_names_with_new_mid():
        path_lmd_v = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/LMD-cleaned_valid/*.mxl'
        ic(len(list(glob.iglob(path_lmd_v))))
        for fnm in glob.iglob(path_lmd_v):
            fnm_new = clean_whitespace(fnm)
            if fnm != fnm_new:
                os.rename(fnm, fnm_new)
                print(f'Renamed {logi(fnm)} => {logi(fnm_new)}')
    # fix_match_mxl_names_with_new_mid()

    def get_lmd_subset():
        # fnms = get_lmd_cleaned_subset_fnms()
        fnms = get_cleaned_song_paths('LMD-cleaned-subset')
        ic(len(fnms), fnms[:20])
    get_lmd_subset()
