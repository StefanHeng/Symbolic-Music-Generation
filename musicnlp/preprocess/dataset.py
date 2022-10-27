import os
import json
from dataclasses import dataclass

from os.path import join as os_join
from typing import List, Tuple, Dict, Callable, Any, Union, Optional, Iterable

import datasets
import torch
from datasets import Dataset, DatasetDict, DatasetInfo

from stefutil import *
import musicnlp.util.music as music_util
from musicnlp.vocab import MusicVocabulary, MusicTokenizer
from musicnlp.preprocess import transform


__all__ = [
    'get_dataset_dir_name', 'DATASET_NAME2MODE2FILENAME',
    'load_songs', 'iter_songs_n_key',
    'get_dataset', 'AugmentedDataset', 'ProportionMixingDataset'
]


logger = get_logger('Dataset')


DATASET_NAME2MODE2FILENAME: Dict[str, Dict[str, str]] = {  # Are in pitch_kind `step`
    # `dataset name` => `mode` => `filename`
    'LMD': {
        'melody': '',
        'full': '22-10-22_Extracted-LMD_{n=176640}_{md=f, prec=5, th=1}'
    },
    'MAESTRO': {
        'melody': '',
        'full': '22-10-22_Extracted-MAESTRO_{n=1276}_{md=f, prec=5, th=1}',
    },
    'POP909': {
        'melody': '',
        'full': '22-10-22_Extracted-POP909_{n=909}_{md=f, prec=5, th=1}'
    }
}


def get_dataset_dir_name(*dnms, mode='full') -> Union[str, List[str]]:
    def _get_single(dnm: str) -> str:
        ca.check_mismatch('Dataset Name', dnm, DATASET_NAME2MODE2FILENAME.keys())
        return get(DATASET_NAME2MODE2FILENAME, f'{dnm}.{mode}')
    if len(dnms) == 1:
        return _get_single(dnms[0])
    else:
        return [_get_single(dnm) for dnm in dnms]


_Song = Union[str, Dict[str, Any]]


def load_songs(*dnms, as_dict: bool = True, as_iter: bool = False) -> Union[List[_Song], Iterable[_Song]]:
    """
    Get individual song `score`s from a JSON `music_export` output
    """
    def _load_single(dnm_):
        logger.info(f'Loading songs from JSON dataset {pl.i(dnm_)}... ')
        with open(os.path.join(music_util.get_processed_path(), f'{dnm_}.json'), 'r') as f:
            dset = json.load(f)

        def gen():
            for s in dset['music']:
                yield s if as_dict else s['score']
        return gen() if as_iter else list(gen())
    if as_iter:
        return chain_its(_load_single(dnm) for dnm in dnms)
    else:
        return sum((_load_single(dnm_) for dnm_ in dnms), start=[])


@dataclass
class IterSongOutput:
    generator: Iterable[Tuple[str, str]] = None
    total: int = None


def iter_songs_n_key(songs: Iterable[Dict[str, Any]]) -> IterSongOutput:
    """
    :param songs: songs, each containing `score` and possible `key`s, per music extraction API
    :return: songs, each with each of its possible key
    """
    n = sum(len(s['keys']) for s in songs) if isinstance(songs, list) else None  # Don't consume it otherwise

    def gen():
        for s in songs:
            txt = s['score']
            for key in s['keys']:
                yield txt, key
    return IterSongOutput(generator=gen(), total=n)


def get_dataset(
        dataset_names: Union[str, List[str]],
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, shuffle_seed: int = None, fast=True, pbar: bool = False,
        splits: Union[str, List[str]] = None
) -> Union[Dataset, DatasetDict]:
    """
    Get dataset preprocessed for training

    If multiple dataset names are given, the datasets are stacked
    """
    def load_single(dnm: str) -> Union[Dataset, DatasetDict]:
        return datasets.load_from_disk(os_join(music_util.get_processed_path(), 'hf', dnm))

    logger.info('Loading dataset from disk... ')
    if isinstance(dataset_names, (list, tuple)):
        dset = [load_single(dnm) for dnm in dataset_names]
        if isinstance(dset[0], DatasetDict):
            def k2concat_args(split) -> Dict:
                dsets = [d_dict[split] for d_dict in dset]
                descs = [json.loads(d_dict[split].info.description) for d_dict in dset]
                descs = {k_: [d[k_] for d in descs] for k_ in descs[0].keys()}  # Merge all keys
                assert list_is_same_elms(descs['extractor_meta']), \
                    f'{pl.i("extractor_meta")} must be the same for all datasets to combine'
                descs['extractor_meta'] = descs['extractor_meta'][0]
                info = DatasetInfo(description=json.dumps(descs))
                return dict(dsets=dsets, info=info)
            dset = DatasetDict(
                {split: datasets.concatenate_datasets(**k2concat_args(split)) for split in dset[0].keys()}
            )
    else:
        dset = load_single(dataset_names)
    if splits is not None:
        if isinstance(splits, str):
            splits = [splits]
        dset = DatasetDict({s: dset[s] for s in splits})
    if n_sample is not None:
        logger.info(f'Sampling the first {pl.i(n_sample)} examples... ')
        if isinstance(dset, Dataset):
            dset = dset.select(range(n_sample))
        else:  # dict
            dset = DatasetDict({k: v.select(range(min(n_sample, len(v)))) for k, v in dset.items()})
    if map_func is not None:
        logger.info(f'Mapping... ')
        n_cpu = os.cpu_count()
        if fast and n_cpu >= 2:
            if not pbar:
                datasets.disable_progress_bar()

        dset = dset.map(map_func, batched=True, remove_columns=remove_columns, num_proc=n_cpu)
        datasets.enable_progress_bar()
    if shuffle_seed:
        logger.info(f'Shuffling with seed {pl.i(shuffle_seed)}... ')
        dset = dset.shuffle(seed=shuffle_seed)
    return dset


class AugmentedDataset:
    """
    A wrapper around a datasets.Dataset, with my custom augmentation about
        1) inserting a likely key
        2) mix-up relative order between melody & bass channels
            Can be one of [`full`, `swap`], where, at uniform random,
                if `full`: individual notes in melody & bass are completely interleaved
                if `swap`: the order of melody & bass are swapped

    For each song, sample one of the potential keys based on confidence
        See `musicnlp.preprocess.key_finder.py`
    """
    def __init__(
            self, dataset: Union[str, Dataset], tokenizer: MusicTokenizer = None,  mode: str = None,
            random_crop: Union[bool, int] = False,
            pitch_kind: str = None, insert_key: bool = False, pitch_shift: bool = False,
            channel_mixup: Union[bool, str] = False, dataset_split: str = None
    ):
        ca(extract_mode=mode)
        ca.check_mismatch('Dataset Split', dataset_split, ['train', 'test'])
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
            assert prec == tokenizer.precision and pitch_kind == tokenizer.pitch_kind
        else:
            self.tokenizer = MusicTokenizer(precision=prec, pitch_kind=pitch_kind)

        self.pitch_kind, self.tmp = pitch_kind, None
        if pitch_kind == 'midi':
            self.tmp = transform.ToMidiPitch()

        vocab = self.tokenizer.vocab

        self.random_crop, self.rc = random_crop, None
        if random_crop:
            cm = random_crop if isinstance(random_crop, int) else 1
            self.rc = transform.RandomCrop(start_of_bar=vocab.start_of_bar, crop_mult=cm, return_as_list=True)

        sr_vocab = vocab if vocab.pitch_kind == 'step' else MusicVocabulary(pitch_kind='step')
        self.sr = transform.SanitizeRare(vocab=sr_vocab, return_as_list=True)  # since input text will be in `step`
        self.ki, self.ps, self.cm = None, None, None
        self.insert_key, self.pitch_shift, self.channel_mixup = insert_key, pitch_shift, channel_mixup
        if insert_key:
            self.ki = transform.KeyInsert(vocab=vocab, return_as_list=True)
        if pitch_shift:
            if not insert_key:
                raise ValueError('A key must be inserted for pitch shifting')
            pk = self.tokenizer.pitch_kind
            if pk != 'degree':
                raise ValueError(f'Tokenization will not work: '
                                 f'Pitch Kind should be {pl.i("degree")} for pitch shifting, but found {pl.i(pk)}')
            self.ps = transform.PitchShift(return_as_list=True)
            assert pitch_kind == 'degree'
        else:
            assert pitch_kind != 'degree'  # incompatible
        if channel_mixup:
            if mode != 'full':
                raise ValueError(f'{pl.i("mix_up")} only works with mode={pl.i("full")}')
            mode = 'full' if isinstance(channel_mixup, bool) else channel_mixup
            self.cm = transform.ChannelMixer(precision=prec, vocab=vocab, mode=mode, return_as_list=True)

        self.dataset_split = dataset_split

    @property
    def meta(self) -> Dict[str, Any]:
        d = dict(pch=self.pitch_kind[0])
        if self.insert_key:
            d['ins-key'] = 'T'
        if self.pitch_shift:
            d['pch-sft'] = 'T'
        if self.channel_mixup:
            d['mix-up'] = self.cm.mode[0]
        return d

    @classmethod
    def from_hf(
            cls, dataset_names: Union[str, List[str]], tokenizer: MusicTokenizer = None,
            get_dataset_args: Dict = None, **kwargs
    ) -> Union['AugmentedDataset', Dict[str, 'AugmentedDataset'], Dataset, DatasetDict]:
        """
        From path(s) to huggingface dataset(s), based on `get_dataset`
        """
        dset = get_dataset(dataset_names, **(get_dataset_args or dict()))
        if isinstance(dset, Dataset):
            return cls(dset, tokenizer, **kwargs)
        else:
            dset: DatasetDict
            return {dnm: cls(ds, tokenizer, **kwargs) for dnm, ds in dset.items()}

    @property
    def info(self) -> DatasetInfo:
        return self.dset.info

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx: int):
        if not isinstance(idx, int):
            raise ValueError('Batched indexing not supported')

        item = self.dset[idx]
        toks = item['score']

        if self.random_crop and self.dataset_split == 'train':  # TODO: Do this for eval too for consistency?
            toks = self.rc(toks)

        toks = self.sr(toks)
        if self.pitch_kind == 'midi':
            toks = self.tmp(toks)

        if self.insert_key:
            toks = self.ki(toks, item['keys'])
        if self.pitch_shift:
            toks = self.ps(toks)
        # if self.channel_mixup and self.dataset_split == 'train':  # No random swapping order in eval
        if self.channel_mixup:  # TODO: too large a difference, not sure if best loss makes sense
            toks = self.cm(toks)
        if isinstance(toks, list):
            toks = ' '.join(toks)

        sanity_check = False
        # sanity_check = True
        if sanity_check:
            ori, new = item['score'], toks
            ori, new = ori[:200], new[:200]
            mic(ori, new)
            raise NotImplementedError
        return self.tokenizer(toks, padding='max_length', truncation=True)


class ProportionMixingDataset:
    """
    Examples-proportional mixing from T5
    TODO: failed to find a pytorch working implementation

    Equivalent to, for the larger datasets, a new subset is taken at each epoch,
        then sample in the joined subset once
    """

    def __init__(self, dataset_list: List[Dataset] = None, k: int = None):
        """
        :param dataset_list: Ordered list of datasets
        :param k: Artificial limit
        """
        self.dsets = dataset_list
        assert k is not None
        self.k = k

        self.dset_szs = [min(len(d), k) for d in self.dsets]
        self.sz = sum(self.dset_szs)

        self._sampled_idxs: List[Optional[torch.Tensor]] = [None] * len(self.dsets)
        self.sample()

        self._info = None

    @property
    def meta(self) -> Dict[str, Any]:
        metas = [(d.meta if hasattr(d, 'meta') else None) for d in self.dsets]
        assert all(metas[0] == m for m in metas)  # sanity check datasets of the same type
        d = metas[0] or dict()
        d['mix-lim'] = self.k
        return d

    @property
    def info(self) -> DatasetInfo:
        if self._info is None:
            descs = []
            for d in self.dsets:
                _desc: str = d.info.description
                descs.append(json.loads(_desc))
                d.info.description = None

            # sanity check, the only difference would be `json_filename` in description
            ds1 = self.dsets[0]
            assert all(ds1.info.description == d.info.description and ds1.info.features == d.info.features for d in
                       self.dsets[1:])

            assert all(set(d.keys()) == {'extractor_meta', 'json_filename'} for d in descs)
            assert all(d['extractor_meta'] == descs[0]['extractor_meta'] for d in descs[1:])
            desc = dict(extractor_meta=descs[0]['extractor_meta'], json_filename=[d['json_filename'] for d in descs])
            self._info = DatasetInfo(description=json.dumps(desc), features=ds1.info.features)
        return self._info

    def sample(self):
        """
        Sub-sample datasets larger than k

        Intended to call in each epoch
        """
        for i, dset in enumerate(self.dsets):
            sz = len(dset)
            if sz > self.k:
                self._sampled_idxs[i] = torch.randperm(sz)[:self.k]

    def __len__(self):
        return self.sz

    def _idx2dset_idx(self, idx: int) -> Tuple[int, int]:
        """
        Convert a global index to a dataset index
        """
        for i, sz in enumerate(self.dset_szs):
            if idx < sz:
                return i, idx
            idx -= sz
        raise ValueError('Should not happen')

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError('Batched indexing not supported')
        idx_dset, idx = self._idx2dset_idx(idx)
        dset = self.dsets[idx_dset]
        if self._sampled_idxs[idx_dset] is not None:  # A sub-sample index
            idx = self._sampled_idxs[idx_dset][idx].item()
        return dset[idx]


if __name__ == '__main__':
    import transformers

    from musicnlp.util import *

    mic.output_width = 512

    seed = sconfig('random-seed')
    transformers.set_seed(seed)  # to test key sampling

    pop, mst, lmd = get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')

    def check_n_song():
        dnms = [pop, mst, lmd]
        for dnm in dnms:
            dsets = get_dataset(dnm)
            mic(dnm, dsets)
    check_n_song()

    def check_combined_dset():
        dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
                  'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-52-41'
        # dset_ = get_dataset(dnm_909)
        # dset_ = get_dataset(dnm_lmd)
        dset_ = get_dataset([dnm_909, dnm_lmd], shuffle_seed=seed)
        mic(dset_)
        mic(dset_['train'][0])

        info_ = json.loads(dset_['train'].info.description)
        mic(info_)
    # check_combined_dset()

    def check_key_sample_data_loading():
        # dnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        dnm = 'musicnlp music extraction, dnm=LMD-cleaned-subset, n=10269, ' \
              'meta={mode=melody, prec=5, th=1}, 2022-04-17_11-52-15'
        tokenizer = MusicTokenizer()
        mic(tokenizer)
        dset = AugmentedDataset.from_hf(dnm, tokenizer=tokenizer)
        tr, vl = dset['train'], dset['test']
        for i in range(16):
            ids = tr[i]['input_ids']
            mic(len(ids), tokenizer.decode(ids)[:100])
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
        mic(dset)
        tr = dset['train']
        mic(tr[:2])
    # check_keys_stored_in_dset()
