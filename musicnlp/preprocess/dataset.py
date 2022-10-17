import os
import json
import random
from os.path import join as os_join
from typing import List, Tuple, Dict, Callable, Union, Optional

import datasets
import torch
from datasets import Dataset, DatasetDict, DatasetInfo

from stefutil import *
import musicnlp.util.music as music_util
from musicnlp.vocab import VocabType, ElmType, Channel, MusicElement, MusicVocabulary, MusicTokenizer
from musicnlp.preprocess.music_converter import MusicConverter


__all__ = [
    'load_songs',
    'DATASET_NAME2MODE2FILENAME', 'get_dataset', 'AugmentedDataset', 'ProportionMixingDataset'
]


DATASET_NAME2MODE2FILENAME: Dict[str, Dict[str, str]] = {
    # `dataset name` => `mode` => `filename`
    'LMD': {
        'melody': '',
        'full': '22-10-03_Extracted-LMD_{n=176640}_{md=f, prec=5, th=1}'
    },
    'MAESTRO': {
        'melody': '22-10-03_Extracted-MAESTRO_{n=1276}_{md=m, prec=5, th=1}',
        'full': '22-10-03_Extracted-MAESTRO_{n=1276}_{md=f, prec=5, th=1}',
    },
    'POP909': {
        'melody': '22-10-03_Extracted-POP909_{n=909}_{md=m, prec=5, th=1}',
        'full': '22-10-03_Extracted-POP909_{n=909}_{md=f, prec=5, th=1}'
    }
}


def load_songs(*dnms) -> List[str]:
    """
    Get individual song `score`s from a JSON `music_export` output
    """
    if not hasattr(load_songs, 'logger'):
        load_songs.logger = get_logger('Load Songs')

    def _load_single(dnm_):
        load_songs.logger.info(f'Loading songs in dataset {pl.i(dnm_)}... ')
        with open(os.path.join(music_util.get_processed_path(), f'{dnm_}.json'), 'r') as f:
            dset = json.load(f)
        return [s['score'] for s in dset['music']]
    return sum((_load_single(dnm_) for dnm_ in dnms), start=[])


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

    logger = get_logger('Get Dataset')
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


class ChannelMixer:
    """
    Reorder notes across channels while keeping the order within the channel

    For each change of channel, prefix it
    """
    e_m, e_b = MusicElement(type=ElmType.melody), MusicElement(type=ElmType.bass)

    def __init__(self, precision: int = 5, vocab: MusicVocabulary = None, mode: str = 'full'):
        self.mc = MusicConverter(mode='full', precision=precision, vocab=vocab)
        self.vocab = self.mc.vocab

        ca(channel_mixup=mode)
        self.mode = mode

    def __call__(self, text: str, return_as_list: bool = False) -> str:
        out = self.mc.str2notes(text, group=True)

        sanity_check = True
        # sanity_check = False
        if sanity_check:
            for elms in out.elms_by_bar:
                mixed = self._mix_up_bar_notes(elms)
                d = self.mc.split_notes(mixed)
                mic(elms, mixed, d)
                recon = [
                    MusicElement(type=ElmType.melody), *d['melody'], MusicElement(type=ElmType.bass), *d['bass']
                ]
                assert elms == recon  # sanity check reconstruction, no info loss
            exit(1)
        ret = self.vocab.music_elm2toks(out.time_sig) + self.vocab.music_elm2toks(out.tempo)
        if out.key:
            ret += self.vocab.music_elm2toks(out.key)
        ret += sum((self._bar_music_elms2str(elms) for elms in out.elms_by_bar), start=[])
        ret += [self.vocab.end_of_song]
        return ret if return_as_list else ' '.join(ret)

    def _bar_music_elms2str(self, elms: List[MusicElement]):
        return [self.vocab.start_of_bar] + sum(
            (self.vocab.music_elm2toks(e) for e in self._mix_up_bar_notes(elms)), start=[]
        )

    @staticmethod
    def _bin_sample() -> bool:
        return random.randint(0, 1) == 0

    def _mix_up_bar_notes(self, elms: List[MusicElement]) -> List[MusicElement]:
        d_notes = self.mc.split_notes(elms)
        notes_m, notes_b = d_notes['melody'], d_notes['bass']

        if self.mode == 'full':
            notes_m, notes_b = iter(notes_m), iter(notes_b)
            elms = []
            note_m, note_b = next(notes_m, None), next(notes_b, None)
            c_cur, c_prev = None, None
            add_mel = None
            while note_m and note_b:
                add_mel = ChannelMixer._bin_sample()
                c_cur = Channel.melody if add_mel else Channel.bass
                diff_c = c_cur != c_prev
                if diff_c:
                    elms.append(ChannelMixer.e_m if add_mel else ChannelMixer.e_b)
                if c_cur == Channel.melody:
                    elms.append(note_m)
                    note_m = next(notes_m, None)
                else:
                    elms.append(note_b)
                    note_b = next(notes_b, None)
                c_prev = c_cur
            assert add_mel is not None  # sanity check
            if note_m:
                if not add_mel:
                    elms.append(ChannelMixer.e_m)
                elms.append(note_m)
                elms.extend(notes_m)
            else:
                if add_mel:
                    elms.append(ChannelMixer.e_b)
                assert note_b
                elms.append(note_b)
                elms.extend(notes_b)
            return elms
        else:  # `swap`
            if ChannelMixer._bin_sample():
                return elms
            else:  # swap at 50% chance
                return [ChannelMixer.e_b, *notes_b, ChannelMixer.e_m, *notes_m]


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
            self, dataset: Union[str, Dataset], tokenizer: MusicTokenizer = None,
            augment_key: bool = False, channel_mixup: Union[bool, str] = False, mode: str = None
    ):
        ca(extract_mode=mode)
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

        self.augment_key = augment_key
        self.channel_mixup = channel_mixup
        if channel_mixup and mode != 'full':
            raise ValueError(f'{pl.i("mix_up")} only works with mode={pl.i("full")}')
        self.cm = None
        if channel_mixup:
            mode = 'full' if isinstance(channel_mixup, bool) else channel_mixup
            self.cm = ChannelMixer(precision=prec, vocab=self.tokenizer.vocab, mode=mode)

    @classmethod
    def from_hf(
            cls, dataset_names: Union[str, List[str]], tokenizer: MusicTokenizer = None,
            get_dataset_kwargs: Dict = None, **kwargs
    ) -> Union['AugmentedDataset', Dict[str, 'AugmentedDataset'], DatasetDict]:
        """
        From path(s) to huggingface dataset(s), based on `get_dataset`
        """
        dset = get_dataset(dataset_names, **(get_dataset_kwargs or dict()))
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
        if self.channel_mixup:
            toks = self.cm(toks, return_as_list=True)
        if self.augment_key:
            if isinstance(toks, list):  # already split into tokens, by `channel_mixup`
                toks = toks
            else:
                toks = toks.split()
            assert self.tokenizer.vocab.type(toks[0]) == VocabType.time_sig  # sanity check data well-formed
            assert self.tokenizer.vocab.type(toks[1]) == VocabType.tempo

            key_tok = self.tokenizer.vocab(pt_sample(item['keys']))[0]
            toks.insert(2, key_tok)
        if isinstance(toks, list):
            toks = ' '.join(toks)
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
