"""
Data Loader for pytorch

obsolete, see `musicnlp.preprocess.datasets.py`
"""

from os.path import join as os_join
import json
import itertools

import numpy as np

import musicnlp.util.music as music_util
from musicnlp.preprocess.melody_extractor import get_tokenizer

tokenizer = get_tokenizer()
ID_PAD = tokenizer['encoder']['[PAD]']


class MelodyLoader:
    # File path for decoded song ids
    SONG_FP = os_join(music_util.get_processed_path(), 'Song-ids.json')

    def __init__(self, pad=True):
        """
        :param pad: If true, instances returned are padded to the maximum sequence length
        """
        self.pad = pad
        with open(MelodyLoader.SONG_FP, 'r') as f:
            songs: list[dict[str]] = json.load(f)
            self.nms = [d['nm'] for d in songs]
            self.ids = np.array(list(itertools.zip_longest(*[d['ids'] for d in songs], fillvalue=ID_PAD))).T

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        ids_ = self.ids[idx]
        return ids_ if self.pad else ids_[ids_ != ID_PAD]  # Remove padding


if __name__ == '__main__':
    from icecream import ic

    ml = MelodyLoader()

    def sanity_check():
        ic(len(ml), ml[0], ml.nms[:20])
        ic(ml[0].shape, MelodyLoader(pad=False)[0].shape)
    sanity_check()

    def why_starting_rests():
        ic(ml[0])
    # why_starting_rests()
