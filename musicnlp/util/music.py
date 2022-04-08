import glob
from shutil import copyfile
from collections import defaultdict

from tqdm import tqdm

from musicnlp.util.util import *


def get_my_example_songs(k=None, pretty=False, fmt='mxl', extracted: bool = False):
    """
    :return: A list of or single MIDI file path
    """
    fmt, formats = fmt.lower(), ['mxl', 'midi']
    assert fmt in formats, f'Invalid format: expected one of {logi(formats)}, got {logi(fmt)}'
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
        fnm='musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-03-01 02-29-29',
        dir_=get_processed_path(),
        k: Union[int, str] = 0
) -> str:
    with open(os.path.join(dir_, f'{fnm}.json')) as f:
        dset = json.load(f)['music']
    if isinstance(k, int):
        return dset[k]['text']
    else:
        return next(d['text'] for d in dset if k in d['title'])


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
        fnms = sorted(glob.iglob(os.path.join(path_ori, d_dset['song_fmt'])))

        # empirically seen as a problem: some files are essentially the same title, ending in different numbers
        # See `ValueError` below
        version_counter = defaultdict(int)

        def path2fnm(p_: str):
            paths_last = p_.split(os.sep)[-2:]
            author, title = paths_last
            my_lim, os_lim = 256-32, 255
            title = stem(title)
            if len(title) > my_lim:
                k_title = title[:my_lim]
                title = f'{k_title}... - v{version_counter[k_title]}'  # ensures no duplicates
                version_counter[k_title] += 1
            fnm_ = f'{author} - {title}'[:255-4]  # the top filename limit
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


def get_lmd_cleaned_subset_fnms() -> List[str]:
    """
    My subset of LMD-cleaned dataset
        MIDI files that can't be converted to MXL via MuseScore are excluded
        Only one unique artist-song is picked among the many versions
            Resolve by just taking the first one

    Expects `convert_dataset` called first
    """
    # TODO: this applies to deprecated version of dataset path & filename, update
    # this folder contains all MIDI files that can be converted to MXL, on my machine
    path = os.path.join(PATH_BASE, DIR_DSET, 'LMD-cleaned_valid')
    # <artist> - <title>(.<version>)?.mid
    pattern = re.compile(r'^(?P<artist>.*) - (?P<title>.*)(\.(?P<version>[1-9]\d*))?\.mid$')
    pattern_title = re.compile(r'((?P<title>.*)\.(?P<version>[1-9]\d*))?')
    d_song2fnms: Dict[Tuple[str, str], Dict[int, str]] = defaultdict(dict)
    fnms = sorted(glob.iglob(os.path.join(path, '*.mid')))
    for fnm in tqdm(fnms, desc='Getting LMD-cleaned subset', unit='song'):
        fnm = stem(fnm, keep_ext=True)
        m = pattern.match(fnm)
        artist, title = m.group('artist'), m.group('title')
        assert artist is not None and title is not None
        m = pattern_title.match(title)
        title_, version = m.group('title'), m.group('version')
        if title_ is None:
            assert version is None
        else:
            title, version = title_, int(version)
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
        path = os.path.join(path, dir_nm, d_dset[fmt])
        return sorted(glob.iglob(path, recursive=True))


if __name__ == '__main__':
    from icecream import ic

    def check_fl_nms():
        dnm = 'POP909'
        fnms = get_cleaned_song_paths(dnm)
        ic(len(fnms), fnms[:20])
        fnms = get_cleaned_song_paths(dnm, fmt='song_fmt_exp')
        ic(len(fnms), fnms[:20])
    # check_fl_nms()

    fl_nms = get_lmd_cleaned_subset_fnms()
    ic(len(fl_nms), fl_nms[:20])