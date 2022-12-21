"""
The reason WordPiece doesn't improve performance could be there's too many ways to encode the same song sequence

To maintain the good accuracy of single token, while keeping a unique tokenization for a given song,
    try to merge the highest-occurring music elements, i.e. single note & tuplets, into a single token
"""

import json
from os.path import join as os_join
from typing import List, Tuple, Dict, Iterable, Union
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.tokenization_utils import PreTrainedTokenizer, BatchEncoding
from tqdm.auto import tqdm

from stefutil import *
from musicnlp.util import *
from musicnlp.vocab import MusicVocabulary, MusicTokenizer
from musicnlp.preprocess import MusicConverter, dataset


__all__ = ['PairMergeTokenizerTrainer', 'PairMergeTokenizer', 'load_pairmerge_tokenizer']


logger = get_logger('Pair-Merge Tokenizer')


class PairMergeTokenizerTrainer:
    def __init__(self, pitch_kind: str = 'step', mode: str = 'full', **kwargs):
        self.pitch_kind = pitch_kind
        self.mode = mode

        self.vocab = MusicVocabulary(pitch_kind=pitch_kind, **kwargs)
        self.aug_key = pitch_kind == 'degree'
        self.lsm = dataset.LoadSongMap(pitch_kind='degree') if self.aug_key else None
        self.mc = MusicConverter(mode=mode)

    def __call__(
            self, dataset_names: List[str] = None, vocab_size: int = None, coverage_ratio: float = None,
            save: Union[bool, str] = None, plot_meta: bool = None, concurrent: Union[bool, Dict] = False
    ):
        """
        :param dataset_names: List of dataset of songs to train on
        :param save: File to save trained tokenizer
        :param vocab_size: Total vocabulary size, initial vocab size + added merged tokens
        :param coverage_ratio: Merged tokens are added until the coverage ratio is reached
        :param plot_meta: Metadata on coverage ratio is plotted
        :param concurrent: If true, song tokens are added concurrently

        .. note:: Only one of `vocab_size` and `coverage_ratio` should be specified
        """
        d_log = dict(pitch_kind=self.pitch_kind, mode=self.mode, vocab_size=vocab_size, coverage_ratio=coverage_ratio)
        logger.info(f'Training w/ {pl.i(d_log)}...')

        if not ((vocab_size or coverage_ratio) and not (vocab_size and coverage_ratio)):
            _args = ['vocab_size', 'coverage_ratio']
            raise ValueError(f'Specify one of {pl.i(_args)}')

        vsz_add = None
        if vocab_size:
            vsz_add = vocab_size - len(self.vocab)

        _songs: List[Dict] = dataset.load_songs(*dataset_names)
        if self.aug_key:
            out = dataset.iter_song(_songs)
            n, songs = out.total, out.generator
        else:
            n, songs = len(_songs), (s['score'] for s in _songs)

        if self.aug_key:
            def gen():
                for s in songs:
                    s, key = s
                    yield self.lsm(text=s, key=key)
        else:
            gen = (s['score'] for s in _songs)

        c = Counter()  # TODO: weigh by dataset size?
        with_tqdm = dict(total=n, desc='Counting music elements')
        if concurrent:
            args = concurrent if isinstance(concurrent, dict) else dict()
            for elms in conc_yield(self._song2uniq_elms, gen(), **args, with_tqdm=with_tqdm):
                c.update(elms)
        else:
            for song in tqdm(gen(), **with_tqdm):
                c.update(self._song2uniq_elms(song))
        n_uniq = len(c)
        logger.info(f'# unique music elements: {pl.i(n_uniq)}')

        counts, ratio = PairMergeTokenizerTrainer._counter2ratio(counter=c)

        if vocab_size:
            if vsz_add > n_uniq:
                logger.info(f'All unique {pl.i(n_uniq)} music elements will be added to the vocabulary, '
                            f'instead of {pl.i(vsz_add)}')
                vsz_add = len(c)
                coverage_ratio = 1
            else:
                coverage_ratio = ratio[vsz_add]
        elif coverage_ratio:
            vsz_add = int(np.searchsorted(ratio, coverage_ratio, side='right'))
            vocab_size = len(self.vocab) + vsz_add
            logger.info(f'Adding {pl.i(vsz_add)} merged elements ')
        d_log = dict(vocab_size=vocab_size, coverage_ratio=coverage_ratio)
        logger.info(f'Final {pl.i(d_log)}')

        save_meta = None
        if save:  # as whole percentage
            save_meta = dict(vsz=vocab_size, r=round(coverage_ratio * 100), n=n, pch=self.pitch_kind[0])
        if plot_meta:
            self._plot(counter=c, counts=counts, ratio=ratio, save=save)

        if save:
            mc = c.most_common(n=vsz_add)
            n_vocab = len(self.vocab)
            tok2id = {tok: i + n_vocab for i, (tok, _) in enumerate(mc)}  # In descending order of frequency
            d = dict(
                added_tok2id=tok2id, n_unique=n_uniq, n_added=vsz_add, occurence_count=dict(mc),
                original_vocab_size=len(self.vocab),
                music_vocab=dict(precison=self.vocab.precision, pitch_kind=self.vocab.pitch_kind)
            )

            fnm = save if isinstance(save, str) else 'PairMerge-Tokenizer'
            date = now(fmt='short-date')
            fnm = f'{date}_{fnm}_{pl.pa(save_meta, pad_float=False)}'
            path = os_join(u.tokenizer_path, f'{fnm}.json')

            path_meta = os_join(u.tokenizer_path, f'{fnm}.json')
            with open(path_meta, 'w') as f:
                json.dump(d, f, indent=4)
            logger.info(f'Tokenizer saved to {pl.i(path)}')

    def _song2uniq_elms(self, song: str) -> List[str]:
        ret = []
        out = self.mc.str2tok_elms(song)
        for elms in out.elms_by_bar:
            for me in elms:
                if me != [self.vocab.start_of_melody] and me != [self.vocab.start_of_bass]:
                    ret.append(' '.join(me))
        return ret

    @staticmethod
    def _counter2ratio(counter: Counter) -> Tuple[np.ndarray, np.ndarray]:
        counts = np.empty(len(counter), dtype=int)
        for i, (k, v) in enumerate(counter.most_common()):
            counts[i] = v

        counts = np.sort(counts)[::-1]
        ratio = np.cumsum(counts) / counts.sum()
        return counts, ratio

    def _plot(self, counter: Counter = None, counts: np.ndarray = None, ratio: np.ndarray = None, save: str = None):
        """
        At which vocab size would it cover many of the music elements?
        """
        idx_1st_tup = None
        for i, (k, v) in enumerate(counter.most_common()):
            if self.vocab.start_of_tuplet in k and idx_1st_tup is None:
                idx_1st_tup = i
        s1 = np.where(ratio > 0.68)[0][0]  # std 1 sigma
        s2 = np.where(ratio > 0.95)[0][0]
        s25 = np.where(ratio > 0.986)[0][0]
        s3 = np.where(ratio > 0.997)[0][0]
        d_log = {
            'ordinal of 1st tuplet': idx_1st_tup,
            'ordinal covering 1 sigma': s1, 'ordinal covering 2 sigma': s2,
            'ordinal covering 2.5 sigma': s25, 'ordinal covering 3 sigma': s3
        }
        logger.info(pl.i(d_log))

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        cs = sns.color_palette(palette='husl', n_colors=8)
        ax1.plot(counts, **LN_KWARGS, c=cs[4])

        ax2.plot(ratio, **LN_KWARGS, c=cs[6])
        ax2.vlines(x=s1, ymin=0, ymax=1, lw=0.4, color=cs[0], label=f'68% at vsz={s1}')
        ax2.vlines(x=s2, ymin=0, ymax=1, lw=0.4, color=cs[1], label=f'95% at vsz={s2}')
        ax2.vlines(x=s25, ymin=0, ymax=1, lw=0.4, color=cs[2], label=f'98.6% at vsz={s25}')
        ax2.vlines(x=s3, ymin=0, ymax=1, lw=0.4, color=cs[3], label=f'99.7% at vsz={s3}')
        ax2.vlines(x=idx_1st_tup, ymin=0, ymax=1, lw=0.4, color=cs[7], label=f'1st tuplet at vsz={idx_1st_tup}')

        ax1.set_title('incremental')
        ax2.set_title('cumulative')

        title = 'Pair-Merge Tokenizer Music Element Coverage'
        plt.suptitle(title)
        fig.supxlabel(f'added vocab size')
        plt.legend()

        if save:
            save_fig(title=f'{title}_{save}')
        else:
            plt.show()


class PairMergeTokenizer(MusicTokenizer, PreTrainedTokenizer):

    def __init__(
            self, added_tok2id: Dict[str, int] = None, precision: int = 5, mode: str = 'full', **kwargs
    ):
        """
        :param tokenizer: A trained WordPiece tokenizer on characters
        """
        init_args = dict(precision=precision, name_or_path=self.__class__.__qualname__) | kwargs
        super().__init__(**init_args)

        self.original_vocab_size = len(self.vocab)
        self.added_vocab_size = len(added_tok2id)
        self.added_tok2id = added_tok2id
        self.added_id2tok = {v: k for k, v in added_tok2id.items()}

        self._id2pchs_inc: Dict[int, List[int]] = dict()  # cache, from each id to pitch if it contains any
        self._id2pchs_exc: Dict[int, List[int]] = dict()
        for i in range(self.vocab_size):
            toks = self._convert_id_to_token(i).split()
            self._id2pchs_inc[i] = super().ids2pitches(toks, include_rest_pitch=True)
            self._id2pchs_exc[i] = super().ids2pitches(toks, include_rest_pitch=False)

        self.mc = MusicConverter(mode=mode)

    @classmethod
    def from_file(cls, fnm: str, output_path: str = u.tokenizer_path, **kwargs):
        path = os_join(output_path, fnm)
        logger.info(f'Loading Pair-Merge Tokenizer from {pl.i(path)}... ')
        with open(f'{path}.json', 'r') as f:
            meta = json.load(f)
        init_args = meta['music_vocab'] | kwargs
        ret = cls(added_tok2id=meta['added_tok2id'], **init_args)

        assert meta['original_vocab_size'] == len(ret.vocab)   # sanity check no new modification to vocab
        return ret

    @property
    def vocab_size(self) -> int:
        return self.original_vocab_size + self.added_vocab_size

    def _tokenize(self, text, **kwargs):
        out = self.mc.str2tok_elms(text)
        ret = [out.time_sig, out.tempo]
        if out.key:
            ret.append(out.key)
        if out.omit:
            ret.append(out.omit)

        ret += chain_its(self._tokenize_bar_elms(elms) for elms in out.elms_by_bar)

        if out.end_of_song:
            ret.append(out.end_of_song)
        return ret

    def _tokenize_bar_elms(self, elms: List[List[str]]) -> Iterable[str]:
        """
        Encode music elements in a bar
        """
        yield self.sob_token
        for me in elms:
            tok_merge = ' '.join(me)
            if tok_merge in self.added_tok2id:
                yield tok_merge
            else:
                for tok in me:
                    yield tok

    def encode(self, text, entire_score: bool = True, **kwargs):
        if isinstance(text, str):
            raise NotImplementedError
        else:
            raise NotImplementedError('TODO')

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, list) and isinstance(token_ids[0], list):
            raise NotImplementedError('Not implemented for iterable input')
        else:
            return ' '.join(self._convert_id_to_token(i) for i in token_ids)

    def _convert_id_to_token(self, index: int) -> str:
        return self.vocab.i2t(index) if index < self.original_vocab_size else self.added_id2tok[index]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.t2i(token) if token in self.vocab else self.added_tok2id[token]

    def ids2pitches(self, ids: Iterable[int], include_rest_pitch: bool = True) -> List[int]:
        i2p = self._id2pchs_inc if include_rest_pitch else self._id2pchs_exc
        return sum([i2p[int(i)] for i in ids], start=[])


def load_pairmerge_tokenizer(
        fnm: str = None, pitch_kind: str = 'step', **kwargs
) -> PairMergeTokenizer:
    if pitch_kind == 'midi':
        raise NotImplementedError
    elif pitch_kind == 'step':
        raise NotImplementedError
    else:
        assert pitch_kind == 'degree'
        # fnm = fnm or '22-12-18_PairMerge-Tokenizer_{dnm=POP&MST}_{vsz=4716, r=95, n=8234, pch=d}'
        fnm = fnm or '22-12-18_PairMerge-Tokenizer_{dnm=all}_{vsz=4642, r=95, n=715891, pch=d}'
    return PairMergeTokenizer.from_file(fnm, pitch_kind=pitch_kind, **kwargs)


class _CheckTrainedSingle:
    """
    Check encoding & decoding are bijective
    """
    def __init__(self, tokenizer: PairMergeTokenizer = None):
        self.tokenizer = tokenizer
        vocab = self.tokenizer.vocab
        self.pitch_kind = vocab.pitch_kind

        self.aug_key = self.pitch_kind == 'degree'
        self.lsm = dataset.LoadSongMap(pitch_kind='degree') if self.aug_key else None

    def __call__(self, text: Union[str, Tuple[str, str]] = None):
        if self.aug_key:
            text, key = text
            text = self.lsm(text=text, key=key)

        ids = self.tokenizer(text).input_ids
        dec = self.tokenizer.decode(ids)
        assert dec == text


if __name__ == '__main__':
    from musicnlp._sample_score import sample_full_degree

    mic.output_width = 128

    md = 'full'
    pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')

    def train():
        aug_key = True
        pch_kd = 'degree' if aug_key else 'step'

        # dnms = [pop]
        # dnms = [pop, mst]
        dnms = [pop, mst, lmd]
        sv = True
        if len(dnms) == 1:
            sv = 'PairMerge-Tokenizer_{dnm=POP909}'
        elif len(dnms) == 2:
            sv = 'PairMerge-Tokenizer_{dnm=POP&MST}'
        elif len(dnms) == 3:
            sv = 'PairMerge-Tokenizer_{dnm=all}'

        pmtt = PairMergeTokenizerTrainer(pitch_kind=pch_kd, mode=md)
        # vsz_arg = dict(vocab_size=8192)
        vsz_arg = dict(vocab_size=None, coverage_ratio=0.95)
        conc = False
        # conc = dict(mode='process')
        pmtt(dataset_names=dnms, **vsz_arg, save=sv, plot_meta=True, concurrent=conc)
    train()

    def check_single_tokenize():
        fnm = '22-12-18_PairMerge-Tokenizer_{dnm=POP&MST}_{vsz=4716, r=95, n=8234, pch=d}'
        tokenizer = PairMergeTokenizer.from_file(fnm)
        inputs = tokenizer(sample_full_degree)
        # mic(inputs.keys())
        ids = inputs.input_ids
        # mic(ids)

        dec = tokenizer.decode(ids)
        mic(dec[:2000])
        assert dec == sample_full_degree
    # check_single_tokenize()

    def check_tokenize_all():
        aug_key = True
        pch_kd = 'degree' if aug_key else 'step'

        fnm = '22-12-18_PairMerge-Tokenizer_{dnm=POP&MST}_{vsz=4716, r=95, n=8234, pch=d}'
        tokenizer = PairMergeTokenizer.from_file(fnm)
        check = _CheckTrainedSingle(tokenizer=tokenizer)

        dnms = [pop]
        _songs: List[Dict] = dataset.load_songs(*dnms)
        if aug_key:
            out = dataset.iter_song(_songs)
            n, songs = out.total, out.generator
        else:
            n, songs = len(_songs), (s['score'] for s in _songs)

        for song in tqdm(songs, total=n, desc='Checking reconstruction'):
            check(song)
    # check_tokenize_all()
    # profile_runtime(check_tokenize_all)
