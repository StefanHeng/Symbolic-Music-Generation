import os
import json
from typing import List, Dict, Callable, Union

import torch
import datasets
from datasets import Dataset, DatasetDict

from musicnlp.util import *
import musicnlp.util.music as music_util
from musicnlp.vocab import VocabType, MusicTokenizer


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


class KeySampleDataset:
    """
    A wrapper around a datasets.Dataset, with my custom augmentation about inserting keys
    """
    def __init__(self, dataset: Union[str, Dataset]):
        if isinstance(dataset, str):
            self.dset = datasets.load_from_disk(os.path.join(music_util.get_processed_path(), 'processed', dataset))
        else:
            self.dset = dataset
        # per Dataset creation from my dictionary representation, the `keys` field is the same dictionary with
        # all possible keys, where the values for bad keys are None
        assert 'keys' in self.dset.column_names
        # ic(self.dset.info)
        self.tokenizer = MusicTokenizer(prec=get(json.loads(self.dset.info.description), 'extractor_meta.precision'))
        self.tokenizer.model_max_length = 1024

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        assert isinstance(idx, int), 'Batched indexing not supported'
        item = self.dset[idx]
        # toks = self.tokenizer(item['score'], padding=False, truncation=False)
        # ic(item['score'])
        toks = self.tokenizer.tokenize(item['score'])
        ic(len(toks))
        # sanity check data well-formed
        assert self.tokenizer.vocab.type(toks[0]) == VocabType.time_sig
        assert self.tokenizer.vocab.type(toks[1]) == VocabType.tempo

        d_keys = {k: v for k, v in item['keys'].items() if v}  # filter out `None`s
        keys, weights = zip(*d_keys.items())
        key = keys[torch.multinomial(torch.tensor(weights), 1, replacement=True).item()]
        key_tok = self.tokenizer.vocab(key)[0]
        # ic(keys, weights, key)
        # key_tok = self.tokenizer.encode(key)
        # assert len(key_tok) == 1
        # key_tok = key_tok[0]
        # ic(key_tok)
        toks.insert(2, key_tok)
        # ic(toks)
        item = self.tokenizer(toks, padding='max_length', truncation=True, is_split_into_words=True)
        # ic(item)
        ic(len(item['input_ids']))
        exit(1)
        return item


if __name__ == '__main__':
    import transformers
    from icecream import ic

    ic.lineWrapWidth = 400

    seed = config('random-seed')
    transformers.set_seed(seed)  # to test key sampling

    def check_combined_dset():
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
    # check_combined_dset()

    def check_key_sample_data_loading():
        dnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        dset = KeySampleDataset(dnm)
        for i in range(4):
            ic(dset[i])
    check_key_sample_data_loading()
