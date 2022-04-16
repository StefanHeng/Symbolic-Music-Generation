import os
import json
from typing import List, Dict, Callable, Union

import datasets
from datasets import Dataset, DatasetDict

from musicnlp.util import *
import musicnlp.util.music as music_util


def get_dataset(
        dataset_names: Union[str, List[str]],
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, shuffle_seed: int = None, fast=True
) -> Union[Dataset, DatasetDict]:
    """
    Get dataset preprocessed for training

    If multiple dataset names are given, the datasets are stacked
    """
    def load_single(dnm: str) -> Union[Dataset, DatasetDict]:
        return datasets.load_from_disk(os.path.join(music_util.get_processed_path(), 'processed', dnm))
    if isinstance(dataset_names, (list, tuple)):
        dset = [load_single(dnm) for dnm in dataset_names]
        if isinstance(dset[0], DatasetDict):
            def k2concat_args(split) -> Dict:
                dsets = [d_dict[split] for d_dict in dset]
                descs = [json.loads(d_dict[split].info.description) for d_dict in dset]
                descs = {k_: [d[k_] for d in descs] for k_ in descs[0].keys()}  # Merge all keys
                assert list_is_same_elms(descs['extractor_meta']), \
                    f'{logi("extractor_meta")} must be the same for all datasets to combine'
                descs['extractor_meta'] = descs['extractor_meta'][0]
                info = datasets.DatasetInfo(description=json.dumps(descs))
                return dict(dsets=dsets, info=info)
            dset = DatasetDict(
                {split: datasets.concatenate_datasets(**k2concat_args(split)) for split in dset[0].keys()}
            )
    else:
        dset = load_single(dataset_names)
    if n_sample is not None:
        if isinstance(dset, Dataset):
            dset = dset.select(range(n_sample))
        else:  # dict
            dset = DatasetDict({k: v.select(range(n_sample)) for k, v in dset.items()})
    if map_func is not None:
        num_proc = None
        n_cpu = os.cpu_count()
        if fast and n_cpu >= 2:
            num_proc = n_cpu // 2
            datasets.disable_progress_bar()

        dset = dset.map(map_func, batched=True, remove_columns=remove_columns, num_proc=num_proc)
        datasets.enable_progress_bar()
    dset = dset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else dset.shuffle()  # will always shuffle
    return dset


if __name__ == '__main__':
    from icecream import ic

    ic.lineWrapWidth = 400

    seed = config('random-seed')

    dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-51-01'
    dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
              'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-52-41'
    # dset_ = get_dataset(dnm_909)
    # dset_ = get_dataset(dnm_lmd)
    dset_ = get_dataset([dnm_909, dnm_lmd], shuffle_seed=seed)
    ic(dset_)
    ic(dset_['train'][0])

    info_ = json.loads(dset_['train'].info.description)
    ic(info_)
