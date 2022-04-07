import datasets

from musicnlp.util import *


def get_dataset(
        dataset_name: str,
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, random_seed: int = None, fast=True
) -> datasets.Dataset:
    """
    Get dataset preprocessed for training
    """
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
    # fix_delete_broken_files()
