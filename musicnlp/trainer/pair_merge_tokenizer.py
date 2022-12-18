"""
The reason WordPiece doesn't improve performance could be there's too many ways to encode the same song sequence

To maintain the good accuracy of single token, while keeping a unique tokenization for a given song,
    try to merge the highest-occurring music elements, i.e. single note & tuplets, into a single token
"""

import json
from os.path import join as os_join
from typing import List, Tuple, Dict, Union
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from stefutil import *
from musicnlp.util import *
from musicnlp.vocab.music_vocab import MusicVocabulary
from musicnlp.preprocess import MusicConverter, dataset


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
            self, dataset_names: List[str] = None, vocab_size: int = 8192, coverage_ratio: float = None,
            save: Union[bool, str] = None, plot: bool = None
    ):
        """
        :param dataset_names: List of dataset of songs to train on
        :param save: File to save trained tokenizer
        :param vocab_size: Total vocabulary size, initial vocab size + added merged tokens
        :param coverage_ratio: Merged tokens are added until the coverage ratio is reached
        :param plot: Metadata on coverage ratio is plotted

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

        c = Counter()  # TODO: weigh by dataset size?
        for song in tqdm(songs, total=n, desc='Counting music elements'):
            if self.aug_key:
                song, key = song
                song = self.lsm(text=song, key=key)
            else:  # TODO: midi pitch
                song = song['score']
            out = self.mc.str2tok_elms(song)

            for elms in out.elms_by_bar:
                for me in elms:
                    if me != [self.vocab.start_of_melody] and me != [self.vocab.start_of_bass]:
                        tok = ' '.join(me)
                        c[tok] += 1
        n_uniq = len(c)
        logger.info(f'# unique music elements: {pl.i(n_uniq)}')

        counts, ratio = None, None
        if plot or coverage_ratio:
            counts, ratio = PairMergeTokenizerTrainer._counter2ratio(counter=c)

        save_meta = dict(vsz=vocab_size, n=n, pch=self.pitch_kind[0]) if save else None
        if plot:
            self._plot(counter=c, counts=counts, ratio=ratio, save=save)

        if vocab_size and vsz_add > n_uniq:
            logger.info(f'All unique {pl.i(n_uniq)} music elements are added to the vocabulary, '
                        f'instead of {pl.i(vsz_add)}')
            vsz_add = len(c)
        elif coverage_ratio:
            vsz_add = np.searchsorted(ratio, coverage_ratio, side='right')
            logger.info(f'Adding {pl.i(vsz_add)} merged elements')

        if save:
            mc = c.most_common(n=vsz_add)
            n_vocab = len(self.vocab)
            tok2id = {tok: i + n_vocab for i, (tok, _) in enumerate(mc)}  # In descending order of frequency
            d = dict(
                added_tok2id=tok2id, n_unique=n_uniq, n_added=vsz_add, occurence_count=dict(mc),
                music_vocab=dict(precison=self.vocab.precision, pitch_kind=self.vocab.pitch_kind)
            )

            fnm = save if isinstance(save, str) else 'PairMerge-Tokenizer'
            date = now(fmt='short-date')
            fnm = f'{date}_{fnm}_{pl.pa(save_meta)}'
            path = os_join(u.tokenizer_path, f'{fnm}.json')

            path_meta = os_join(u.tokenizer_path, f'{fnm}_meta.json')
            with open(path_meta, 'w') as f:
                json.dump(d, f, indent=4)
            logger.info(f'Tokenizer saved to {pl.i(path)}')

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
        s3 = np.where(ratio > 0.997)[0][0]
        d_log = {
            'ordinal of 1st tuplet': idx_1st_tup,
            'ordinal covering 1 sigma': s1, 'ordinal covering 2 sigma': s2, 'ordinal covering 3 sigma': s3
        }
        logger.info(pl.i(d_log))

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        cs = sns.color_palette(palette='husl', n_colors=7)
        ax1.plot(counts, **LN_KWARGS, c=cs[3])

        ax2.plot(ratio, **LN_KWARGS, c=cs[5])
        ax2.vlines(x=s1, ymin=0, ymax=1, lw=0.4, color=cs[0], label=f'68% at vsz={s1}')
        ax2.vlines(x=s2, ymin=0, ymax=1, lw=0.4, color=cs[1], label=f'95% at vsz={s2}')
        ax2.vlines(x=s3, ymin=0, ymax=1, lw=0.4, color=cs[2], label=f'99.7% at vsz={s3}')
        ax2.vlines(x=idx_1st_tup, ymin=0, ymax=1, lw=0.4, color=cs[6], label=f'1st tuplet at vsz={idx_1st_tup}')

        ax1.set_title('incremental')
        ax2.set_title('cumulative')

        title = 'Pair-Merge Tokenizer Music Element Coverage'
        plt.suptitle(title)
        fig.supxlabel(f'added vocab size')
        plt.legend()

        if save:
            save_fig(title=f'{title}_{pl.pa(save)}')
        else:
            plt.show()


if __name__ == '__main__':

    md = 'full'
    pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')

    def check_high_occur():
        aug_key = True
        pch_kd = 'degree' if aug_key else 'step'
        lsm = dataset.LoadSongMap(pitch_kind=pch_kd) if aug_key else None

        dnms = [pop]
        # dnms = [pop, mst]
        # dnms = [pop, mst, lmd]

        pmtt = PairMergeTokenizerTrainer(pitch_kind=pch_kd, mode=md)
        pmtt(dataset_names=dnms, vocab_size=8192, save=True, plot=True)
    check_high_occur()
