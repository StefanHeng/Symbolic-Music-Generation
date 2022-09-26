import os
import json
import random
from os.path import join as os_join
from typing import List, Dict, Callable, Union

import datasets
from datasets import Dataset, DatasetDict

from stefutil import *
import musicnlp.util.music as music_util
from musicnlp.vocab import VocabType, ElmType, Channel, MusicElement, MusicTokenizer
from musicnlp.preprocess.music_converter import MusicConverter


def load_songs(*dnms) -> List[str]:
    """
    Get individual song `score`s from a JSON `music_export` output
    """
    if not hasattr(load_songs, 'logger'):
        load_songs.logger = get_logger('Load Songs')

    def _load_single(dnm_):
        load_songs.logger.info(f'Loading songs in dataset {logi(dnm_)}... ')
        with open(os.path.join(music_util.get_processed_path(), f'{dnm_}.json'), 'r') as f:
            dset = json.load(f)
        return [s['score'] for s in dset['music']]
    return sum((_load_single(dnm_) for dnm_ in dnms), start=[])


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
        logger.info(f'Sampling the first {logi(n_sample)} examples... ')
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
        logger.info(f'Shuffling with seed {logi(shuffle_seed)}... ')
        dset = dset.shuffle(seed=shuffle_seed)
    return dset


class AugmentedDataset:
    """
    A wrapper around a datasets.Dataset, with my custom augmentation about
        1) inserting a likely key
        2) mix-up relative order between melody & bass

    For each song, sample one of the potential keys based on confidence
        See `musicnlp.preprocess.key_finder.py`
    """
    def __init__(
            self, dataset: Union[str, Dataset], tokenizer: MusicTokenizer = None,
            augment_key: bool = False, mix_up: bool = False, mode: str = None
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
        self.mix_up = mix_up
        if mix_up and mode != 'full':
            raise ValueError(f'{logi("mix_up")} only works with mode={logi("full")}')
        self.mc = MusicConverter(mode=mode, precision=prec)

    @classmethod
    def from_hf(
            cls, dataset_names: Union[str, List[str]], tokenizer: MusicTokenizer = None,
            get_dataset_kwargs: Dict = None, **kwargs
    ) -> Union['AugmentedDataset', Dict[str, 'AugmentedDataset']]:
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
    def info(self) -> datasets.DatasetInfo:
        return self.dset.info

    def __len__(self):
        return len(self.dset)

    @staticmethod
    def _bin_sample() -> bool:
        return random.randint(0, 1) == 0

    def _mix_up_bar_notes(self, elms: List[MusicElement]):
        """
        Reorder notes across channels while keeping the order within the channel

        For each change of channel, prefix it
        """
        d_notes = self.mc.split_notes(elms)
        notes_m, notes_b = iter(d_notes['melody']), iter(d_notes['bass'])
        elms = []
        note_m, note_b = next(notes_m, None), next(notes_b, None)
        c_cur, c_prev = None, None
        add_mel = None
        e_m, e_b = MusicElement(type=ElmType.melody), MusicElement(type=ElmType.bass)
        while note_m and note_b:
            add_mel = AugmentedDataset._bin_sample()
            c_cur = Channel.melody if add_mel else Channel.bass
            diff_c = c_cur != c_prev
            if diff_c:
                elms.append(e_m if add_mel else e_b)
            if c_cur == Channel.melody:
                elms.append(note_m)
                note_m = next(notes_m, None)
            else:
                elms.append(note_b)
                note_b = next(notes_b, None)
            c_prev = c_cur
                # if c_cur == Channel.melody:
                #     elms.append(note_m)
                #     note_m = next(notes_m, None)
                # else:
                #     elms.append(note_b)
                #     note_b = next(notes_b, None)
                # c_prev = c_cur
        assert add_mel is not None  # sanity check
        if note_m:
            if not add_mel:
                elms.append(e_m)
            elms.append(note_m)
            elms.extend(notes_m)
        else:
            if add_mel:
                elms.append(e_b)
            assert note_b
            elms.append(note_b)
            elms.extend(notes_b)
        # mic(elms)
        # if use_mel:
        #     elms.append(note_m)
        #     note_m = next(notes_m, None)
        # else:
        #     elms.append(note_b)
        #     note_b = next(notes_b, None)
        # if note_m:
        #     while note_m:
        #         elms.append(note_m)
        #         note_m = next(notes_m, None)
        # else:
        #     assert note_b
        #     while note_b:
        #         elms.append(note_b)
        #         note_b = next(notes_b, None)
        # mic(elms)
        # exit(1)
        return elms

    def __getitem__(self, idx: int):
        if not isinstance(idx, int):
            raise ValueError('Batched indexing not supported')

        item = self.dset[idx]
        text = item['score']
        if self.mix_up:
            # mic(text)
            out = self.mc.str2notes(text, group=True)
            # mic(out)
            # mic(self._mix_up_bar_notes(out.elms_by_bar[0]))
            for elms in out.elms_by_bar:
                out = self._mix_up_bar_notes(elms)
                d = self.mc.split_notes(out)
                mic(elms, out, d)
                recon = [MusicElement(type=ElmType.melody), *d['melody'], MusicElement(type=ElmType.bass), *d['bass']]
                assert elms == recon  # sanity check reconstruction, no info loss
            exit(1)
        if self.augment_key:
            toks = text.split()
            assert self.tokenizer.vocab.type(toks[0]) == VocabType.time_sig  # sanity check data well-formed
            assert self.tokenizer.vocab.type(toks[1]) == VocabType.tempo

            key_tok = self.tokenizer.vocab(pt_sample(item['keys']))[0]
            toks.insert(2, key_tok)
            text = ' '.join(toks)
        return self.tokenizer(text, padding='max_length', truncation=True)


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
        dset = AugmentedDataset.from_hf(dnm, tokenizer=tokenizer)
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
