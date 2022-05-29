import os
import json
from os.path import join as os_join
from typing import List, Dict, Callable, Union

import datasets
from datasets import Dataset, DatasetDict

from stefutil import *
import musicnlp.util.music as music_util
from musicnlp.vocab import VocabType, MusicTokenizer


def get_dataset(
        dataset_names: Union[str, List[str]],
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, shuffle_seed: int = None, fast=True, pbar: bool = False
) -> Union[Dataset, DatasetDict]:
    """
    Get dataset preprocessed for training

    If multiple dataset names are given, the datasets are stacked
    """
    def load_single(dnm: str) -> Union[Dataset, DatasetDict]:
        return datasets.load_from_disk(os_join(music_util.get_processed_path(), 'hf', dnm))
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
            dset = DatasetDict({k: v.select(range(min(n_sample, len(v)))) for k, v in dset.items()})
    if map_func is not None:
        num_proc = None
        n_cpu = os.cpu_count()
        if fast and n_cpu >= 2:
            if not pbar:
                datasets.disable_progress_bar()

        dset = dset.map(map_func, batched=True, remove_columns=remove_columns, num_proc=n_cpu)
        datasets.enable_progress_bar()
    if shuffle_seed:
        dset = dset.shuffle(seed=shuffle_seed)
    # else, don't shuffle
    return dset


class KeySampleDataset:
    """
    A wrapper around a datasets.Dataset, with my custom augmentation about inserting keys

    For each song, sample one of the potential keys based on confidence
        See `musicnlp.preprocess.key_finder.py`
    """
    def __init__(self, dataset: Union[str, Dataset], tokenizer: MusicTokenizer = None):
        if isinstance(dataset, str):
            self.dset = datasets.load_from_disk(os_join(music_util.get_processed_path(), 'processed', dataset))
        else:
            self.dset = dataset
        # per Dataset creation from my dictionary representation, the `keys` field is the same dictionary with
        # all possible keys, where the values for bad keys are None
        assert 'keys' in self.dset.column_names  # sanity check
        prec = get(json.loads(self.dset.info.description), 'extractor_meta.precision')
        if tokenizer:
            self.tokenizer = tokenizer
            assert prec == tokenizer.precision
        else:
            self.tokenizer = MusicTokenizer(precision=prec)

    @classmethod
    def from_hf(
            cls, dataset_names: Union[str, List[str]], tokenizer: MusicTokenizer = None,
            get_dataset_kwargs: Dict = None
    ) -> Union['KeySampleDataset', Dict[str, 'KeySampleDataset']]:
        """
        From path(s) to huggingface dataset(s), based on `get_dataset`
        """
        dset = get_dataset(dataset_names, **(get_dataset_kwargs or dict()))
        return cls(dset, tokenizer) if isinstance(dset, Dataset) else {k: cls(v, tokenizer) for k, v in dset.items()}

    @property
    def info(self) -> datasets.DatasetInfo:
        return self.dset.info

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        assert isinstance(idx, int), 'Batched indexing not supported'
        item = self.dset[idx]
        toks = self.tokenizer.tokenize(item['score'])
        assert self.tokenizer.vocab.type(toks[0]) == VocabType.time_sig  # sanity check data well-formed
        assert self.tokenizer.vocab.type(toks[1]) == VocabType.tempo

        key_tok = self.tokenizer.vocab(pt_sample(item['keys']))[0]
        toks.insert(2, key_tok)

        return self.tokenizer(toks, padding='max_length', truncation=True, is_split_into_words=True)


if __name__ == '__main__':
    import transformers
    from icecream import ic

    from musicnlp.util import *

    ic.lineWrapWidth = 512

    seed = sconfig('random-seed')
    transformers.set_seed(seed)  # to test key sampling

    def check_combined_dset():
        dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
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
        # dnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        dnm = 'musicnlp music extraction, dnm=LMD-cleaned-subset, n=10269, ' \
              'meta={mode=melody, prec=5, th=1}, 2022-04-17_11-52-15'
        tokenizer = MusicTokenizer()
        ic(tokenizer)
        dset = KeySampleDataset.from_hf(dnm, tokenizer=tokenizer)
        tr, vl = dset['train'], dset['test']
        for i in range(16):
            ids = tr[i]['input_ids']
            ic(len(ids), tokenizer.decode(ids)[:100])
    # check_key_sample_data_loading()

    def check_keys_stored_in_dset():
        """
        For vanilla training when key isn't passed in, also want to get key of the song to monitor IKR

        Explore how to pass key in
        """
        dnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        path = os_join(music_util.get_processed_path(), 'hf', dnm)
        os.listdir(path)
        dset = datasets.load_from_disk(path)
        ic(dset)
        tr = dset['train']
        ic(tr[:2])
    # check_keys_stored_in_dset()



