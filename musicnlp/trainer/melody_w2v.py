"""
Exploring word2vec on time-slot melody representation

*obsolete*
"""
import os
import datetime
from typing import Iterable

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import torch.nn as nn

from musicnlp.util import *
from .melody_loader import MelodyLoader
from musicnlp.preprocess import MelodyTokenizer


class PitchEmbeddingModel:
    """
    Learn embeddings for each pitch using `word2vec`.

    Note that a major difference with the NLP domain is that vocabulary size
    TODO: what if including duration in each token?
    """
    D_ARCHI = dict(  # Per `Word2Vec` doc
        sg=1,
        cb=0
    )
    D_ALGO = dict(
        hs=1,
        ns=0
    )

    class EpochLogger(CallbackAny2Vec):
        """
        Inspired by
        [Chord-Embeddings.chord2vec](https://github.com/MichiganNLP/Chord-Embeddings/blob/main/Generate/chord2vec.py)
        """
        def __init__(self):
            self.epoch = 1
            self.strt = None
            self.end = None

        def on_epoch_begin(self, model):
            self.strt = datetime.datetime.now()
            log(f'Beginning Epoch #{logi(self.epoch)}... ')

        def on_epoch_end(self, model):
            self.end = datetime.datetime.now()
            log(f'Epoch #{logi(self.epoch)} completed in {logi(fmt_time(self.end - self.strt))} ')
            self.epoch += 1

    def __init__(self, archi='sg', algo='hs', w2v_kwargs=None):
        """
        :param archi: word2vec architecture,
            one of [`sg`, `cb`] for skip-gram or contiguous back of words hierarchical
        :param algo: word2vec training algorithm,
            one of [`hs`, `ns`] for hierarchical softmax or negative sampling
        """
        self.archi = archi
        self.algo = algo

        assert w2v_kwargs is not None
        assert 'sg' not in w2v_kwargs and 'hs' not in w2v_kwargs
        assert 'epochs' in w2v_kwargs
        self.kwargs = dict(  # TODO: UNK & min_count?; Learning rate alpha?
            sg=PitchEmbeddingModel.D_ARCHI[self.archi], hs=PitchEmbeddingModel.D_ALGO[self.algo],
            vector_size=2**6, window=10, sample=1e-1,  # Considering vocabulary size of 100+
            sorted_vocab=1, compute_loss=True,  # TODO: shrink_windows?
            callbacks=[PitchEmbeddingModel.EpochLogger()],
            workers=os.cpu_count(), seed=config('random_seed')
        ) | w2v_kwargs
        self.model: Word2Vec

    def __call__(self, sents: Iterable[Iterable]):
        """
        Trains the word2vec model

        :param sents: Iterable of (tokenized) training samples
        """
        self.model = Word2Vec(sents, **self.kwargs)


class MelodyModel:
    def __init__(self):
        self.model = nn.Transformer(
            d_model=64,
            nhead=8, num_encoder_layers=4, num_decoder_layers=4,
            dim_feedforward=64 * 4, activation='gelu',
            norm_first=False  # As in the Attention paper
        )


if __name__ == '__main__':
    from stefutil import mic
    # mic(type(common_texts), len(common_texts), common_texts[:30])

    ml = MelodyLoader(pad=False)
    mt = MelodyTokenizer()

    def check_as_iter():
        song_it = iter(mt.decode(ids, return_joined=False) for ids in ml)
        # mic(song_it, type(next(song_it)))

        lst = list(song_it)
        mic(len(lst), lst[0][:16])
    # check_as_iter()

    pem = PitchEmbeddingModel(w2v_kwargs=dict(epochs=4))

    def train():
        pem([mt.decode(ids, return_joined=False) for ids in ml])  # To string tokens to apply `Word2Vec`
        model = pem.model
        # for v in model.wv:
        #     mic(v)
        vects = model.wv
        mic(vects, type(vects), len(vects))
        vocab = list(vects.key_to_index)  # Dict from word str to index in embedding mat
        mic(len(vocab), vocab[:20])
        for wd in vocab:
            vec = vects[wd]
            mic(wd, vec[:2], vec.shape)
    # train()
