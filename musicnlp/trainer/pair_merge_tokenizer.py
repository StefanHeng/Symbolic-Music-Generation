"""
The reason WordPiece doesn't improve performance could be there's too many ways to encode the same song sequence

To maintain the good accuracy of single token, while keeping a unique tokenization for a given song,
    try to merge the highest-occurring music elements, i.e. single note & tuplets, into a single token
"""

from typing import List, Dict

from stefutil import *
from musicnlp.vocab.music_vocab import MusicVocabulary
from musicnlp.preprocess import MusicConverter, transform, dataset


class PairMergeTokenizerTrainer:
    def __init__(self, vocab: MusicVocabulary):
        pass


if __name__ == '__main__':
    from collections import Counter

    from tqdm.auto import tqdm

    md = 'full'
    pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')

    def check_high_occur():
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        aug_key = False

        dnms = [pop]
        # dnms = [pop, mst, lmd]
        songs: List[Dict] = dataset.load_songs(*dnms)

        vocab = MusicVocabulary(pitch_kind='step')
        mc = MusicConverter(mode=md, vocab_step=vocab)

        c = Counter()
        for song in tqdm(songs):
            song = song['score']
            out = mc.str2tok_elms(song)
            # mic(out)

            for elms in out.elms_by_bar:
                for me in elms:
                    if me != [vocab.start_of_melody] and me != [vocab.start_of_bass]:
                        tok = ' '.join(me)
                        c[tok] += 1
            # mic(c)
            # raise NotImplementedError
        mic(c.most_common(n=100))
        mic(len(c))

        # At which vocab size would it cover many of the music elements?
        counts = np.empty(len(c), dtype=int)
        for i, (k, v) in enumerate(c.most_common()):
            counts[i] = v
        counts = np.sort(counts)[::-1]
        ratio = np.cumsum(counts) / counts.sum()
        mic(counts.shape, ratio.shape)

        # plt.figure()
        # ax = sns.ecdfplot(counts)
        # q = 99.7
        # ma = round(np.percentile(counts, q=q))
        # mic(ma)
        # ax.set_xlim([0, ma])
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        mic(ax1, ax2)

        r1 = np.where(ratio > 0.68)[0][0]  # std 1 sigma
        r2 = np.where(ratio > 0.95)[0][0]
        r3 = np.where(ratio > 0.997)[0][0]
        mic(r1, r2, r3)

        cs = sns.color_palette(palette='husl', n_colors=7)
        ax1.plot(counts, **LN_KWARGS, c=cs[3])

        ax2.plot(ratio, **LN_KWARGS, c=cs[5])
        ax2.vlines(x=r1, ymin=0, ymax=1, lw=0.4, color=cs[0], label=f'68% at vsz={r1}')
        ax2.vlines(x=r2, ymin=0, ymax=1, lw=0.4, color=cs[1], label=f'95% at vsz={r2}')
        ax2.vlines(x=r3, ymin=0, ymax=1, lw=0.4, color=cs[2], label=f'99.7% at vsz={r3}')

        # ma = counts.max()
        # gap = ma * 0.05
        # ax1.set_ylim([-gap, ma + gap])
        # ax2.set_ylim([-0.05, 1.05])

        ax1.set_title('incremental')
        ax2.set_title('cumulative')

        plt.suptitle('Music Element Coverage')
        fig.supxlabel(f'added vocab size')
        plt.legend()
        plt.show()
    check_high_occur()
