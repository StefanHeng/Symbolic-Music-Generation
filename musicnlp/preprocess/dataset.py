import datasets

from musicnlp.util import *


def get_dataset(
        dataset_names: Union[str, List[str]],
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, random_seed: int = None, fast=True
) -> Union[datasets.Dataset, datasets.DatasetDict]:
    """
    Get dataset preprocessed for training

    If multiple dataset names are given, the datasets are stacked
    """
    def load_single(dnm: str) -> Union[datasets.Dataset, datasets.DatasetDict]:
        return datasets.load_from_disk(os.path.join(get_processed_path(), 'processed', dnm))
    if isinstance(dataset_names, (list, tuple)):
        dset = [load_single(dnm) for dnm in dataset_names]
        if isinstance(dset[0], datasets.DatasetDict):
            dset = datasets.DatasetDict({
                k: datasets.concatenate_datasets([d_dict[k] for d_dict in dset]) for k in dset[0].keys()
            })
    else:
        dset = load_single(dataset_names)
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

    dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-03-01 02-29-29'
    dnm_lmd = 'musicnlp music extraction, n=2210, meta={mode=melody, prec=5, th=1}, 2022-04-10_00-43-03'
    # dset_ = get_dataset(dnm_909)
    dset_ = get_dataset(dnm_lmd)
    # dset_ = get_dataset([dnm_909, dnm_lmd])
    ic(dset_)
    ic(dset_['train'][0])
