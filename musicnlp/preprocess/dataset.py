from shutil import copyfile
from collections import defaultdict

from tqdm import tqdm
import datasets

from musicnlp.util import *


def convert_dataset(dataset_name: str = 'POP909'):
    """
    Convert original dataset to my own format
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
        ic(len(fnms))

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
                t_ = title = f'{k_title}... - v{version_counter[k_title]}'  # ensures no duplicates
                version_counter[k_title] += 1
            fnm_ = f'{author} - {title}'[:255-4]  # the top filename limit
            fnm_ = f'{fnm_}.mid'
            assert len(fnm_) <= os_lim
            return fnm_
        fnms_written = set()
        for p in tqdm(fnms, desc=f'Converting {dataset_name}', unit='song'):
            fnm = path2fnm(p)
            if dnm in fnms_written:
                raise ValueError(f'Duplicate file name because of truncation: path {logi(p)} modified to {logi(fnm)}')
            fnms_written.add(fnm)
            copyfile(p, os.path.join(path_exp, fnm))
        assert len(fnms_written) == len(fnms)


def get_dataset(
        dataset_name: str,
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, random_seed: int = None, fast=True
) -> datasets.Dataset:
    # TODO: only training split?
    dset = datasets.load_from_disk(os.path.join(get_processed_path(), 'hf_datasets', dataset_name))
    if n_sample is not None:
        dset = dset.select(range(n_sample))
    if map_func is not None:
        num_proc = None
        n_cpu = os.cpu_count()
        if fast and n_cpu >= 2:
            num_proc = n_cpu // 2
            datasets.set_progress_bar_enabled(False)

        dset = dset.map(map_func, batched=True, remove_columns=remove_columns, num_proc=num_proc)
        datasets.set_progress_bar_enabled(True)
    dset = dset.shuffle(seed=random_seed) if random_seed is not None else dset.shuffle()
    return dset


if __name__ == '__main__':
    from icecream import ic

    dnm = 'LMD-cleaned'
    # convert_dataset(dnm)

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
        path_broken = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/broken/LMD-cleaned/*.mid'
        set_broken = set(stem(fnm) for fnm in glob.iglob(path_broken))
        ic(set_broken)
        path_lmd_c = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/LMD-cleaned/*.mid'
        for fnm in glob.iglob(path_lmd_c):
            if stem(fnm) in set_broken:
                os.remove(fnm)
                print('Deleted', fnm)
    fix_delete_broken_files()
