from typing import List, Union
from itertools import filterfalse
from collections import Counter

import numpy as np
import music21

from musicnlp.util import *
from musicnlp.util.train import PT_LOSS_PAD
from musicnlp.vocab import VocabType, MusicTokenizer
from musicnlp.vocab.elm_type import *


class IkrMetric:
    """
    Vectorized metric of matched keys per pitch, based on `_get_off_key_ratio`
    """

    def __init__(self, tokenizer: MusicTokenizer, n_init_bars: int = 4):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.n_init_bars = n_init_bars

    def __call__(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """
        Arguments should be batched autoregressive transformer input & output tokens of the same shape in 2D
        """
        assert preds.shape == labels.shape, \
            f'Input and label shapes do not match, {logi(preds.shape)} vs {logi(labels.shape)}'
        ikrs = []
        for pred, label in zip(preds, labels):
            key_tok = label[2]  # expect labels to be well-formed
            assert self.vocab.type(key_tok) == VocabType.key, \
                f'Expect key token at 3rd position of label, got {logi(key_tok)}'
            ikrs.append(self.get_in_key_ratio(pred[label != PT_LOSS_PAD], self.vocab.compact(key_tok)))
        return np.array(ikrs).mean()

    def get_init_key_est(self, gt_token_seq: Union[str, List[str]]):
        tok_lst = gt_token_seq.split() if isinstance(
            gt_token_seq, str) else gt_token_seq

        # Heuristics to determine starting bar
        bar_idx = [idx for idx, tok in enumerate(
            tok_lst) if tok == self.vocab.start_of_bar]
        assert len(bar_idx) > self.n_init_bars + 1, \
            f'Not enough bars for key estimation: expect at least {logi(self.n_init_bars + 1)} total bars in music, ' \
            f'got {logi(len(bar_idx))}'

        pitch_lst = list(
            filterfalse(lambda x: self.vocab.type(x) !=
                        VocabType.pitch, tok_lst[:bar_idx[self.n_init_bars]])
        )
        key_cls = [music21.pitch.Pitch(
            midi=self.vocab.compact(p)).pitchClass for p in pitch_lst]
        key_est = Counter(key_cls).most_common()[0][0]
        return key_est

    def get_pred_stats_pitch(self, preds: List[int]):
        """
        Helper function for sanity check.
        Given a single song, print out relevant statistics of its pitches.
        Args:
            preds (List[int]): List of tokens (represented in ints)
        """
        tok_lst = preds.split() if isinstance(preds, str) else preds
        pitch_lst = list(filterfalse(
            lambda x: self.vocab.type(x) != VocabType.pitch, tok_lst))
        stats_pitch_cls_int = Counter([music21.pitch.Pitch(
            midi=self.vocab.compact(p)).pitchClass for p in pitch_lst])
        stats_pitch_cls_str = Counter([music21.pitch.Pitch(
            midi=self.vocab.compact(p)).name for p in pitch_lst])
        ic(stats_pitch_cls_str)
        ic(stats_pitch_cls_int)

    def get_in_key_ratio(
            self, preds: List[int], key: Key,
            enable_heuristic: bool = False, heuristic_thres: int = 5
    ) -> float:

        tok_lst = preds.split() if isinstance(preds, str) else preds
        target_key = key
        pitch_lst = list(filterfalse(
            lambda x: self.vocab.type(x) != VocabType.pitch, tok_lst))
        num_toks = len(pitch_lst)
        if num_toks == 0:  # No pitch found, assume every pitch is off-note
            return 0
        # Process the given key
        key_type, key_name = key_enum2tuple[target_key]
        # Extract midi values for all available pitches
        pitch_midi = np.array([self.vocab.compact(p) for p in pitch_lst])
        key_offset = key_offset_dict[key_name]
        pred_offset = ((pitch_midi % 12) - key_offset) % 12
        in_key_lst = list(filterfalse(
            lambda x: x in OFFKEY_OFFSET[key_type], pred_offset))
        in_key_ratio = len(in_key_lst) / num_toks
        # Heuristics (Naive implementation)
        # The first pitch of the bar decides the key of the bar
        # TODO: change the processing procedure to speed up
        # Heuristic Update Rule #1: Too many problems + low performance, discard for now
        # pitch_bar_lst = list(filterfalse(
        #     lambda x: self.vocab.type(x) != VocabType.pitch and x != self.tokenizer.sob_token_id, tok_lst))
        # # ic(pitch_bar_lst)
        # # ic(Counter(pitch_bar_lst))
        # s_bar = [idx for idx, tok in enumerate(
        #     pitch_bar_lst) if tok == self.tokenizer.sob_token_id]
        # # ic(s_bar)
        # e_bar, num_bars = s_bar[1:] + [len(pitch_bar_lst)], len(s_bar)
        # ic([pitch_bar_lst[s_bar[i]+1:e_bar[i]]
        #     for i in range(num_bars)])
        # pitch_lst_per_bar = list(filterfalse(lambda x: len(x) <= 1, [pitch_bar_lst[s_bar[i]+1:e_bar[i]]
        #                                                              for i in range(num_bars)]))

        # Heuristic Update method 2: Discard for now
        # for p in pitch_lst:
        #     p_cls = music21.pitch.Pitch(midi=self.vocab.compact(p)).pitchClass
        #     if p_cls != target_key:
        #         num_off_key += 1
        #         # TODO: add cross-bar detection
        #         if prev_p_cls is not None and p_cls == prev_p_cls:
        #             key_count += 1
        #             if key_count >= key_thres:
        #                 target_key = p_cls
        #                 key_count = 0
        #         prev_p_cls = p_cls
        #     else:
        #         prev_p_cls = None
        return in_key_ratio


if __name__ == '__main__':
    import os
    import json

    from tqdm import tqdm
    from icecream import ic

    import musicnlp.util.music as music_util
    from musicnlp.preprocess import MusicExtractor, KeyFinder

    song_nm = 'Merry Go Round of Life'
    # song_nm = '平凡之路'

    def write_eg_song_json(song_name: str = 'Merry Go Round of Life'):
        fnm = music_util.get_my_example_songs(song_name, fmt='MXL')
        me = MusicExtractor()
        # exp = 'str_join'
        exp = 'id'
        score = me(fnm, exp=exp)
        with open(os.path.join(music_util.get_processed_path(), f'{song_name}.json'), 'w') as f:
            json.dump(dict(score=score), f, indent=2)
    # write_eg_song_json(song_nm)

    def get_eg_song_key(song_name: str = 'Merry Go Round of Life'):
        fnm = music_util.get_my_example_songs(song_name, fmt='MXL')

        kf = KeyFinder(fnm)
        keys = kf.find_key(return_type="enum")
        ic(keys)
        # ic(kf.find_scale_degrees(keys))
    # get_eg_song_key(song_nm)

    im = IkrMetric(MusicTokenizer(), n_init_bars=2)

    def check_key_metric():
        # text = music_util.get_extracted_song_eg(
        # k='平凡之路')  # this one has tuplets
        with open(os.path.join(music_util.get_processed_path(), f'{song_nm}.json'), 'r') as f:
            text = json.load(f)['score']
        # Get keys from Key_Finder
        fnm = music_util.get_my_example_songs(song_nm, fmt='MXL')
        kf = KeyFinder(fnm)
        keys = kf.find_key(return_type="enum")
        # ic(text[:200])
        # ic(im.get_off_key_ratio(text, Key.AfMaj))
        exp_out = [im.get_in_key_ratio(text, key) for key in keys]
        ic(exp_out)
        ic(f"Average IKR for {song_nm}: {np.round(np.mean(exp_out), 5)}")
        # ic(im.get_in_key_ratio(text, Key.DMin))
    # check_key_metric()

    def check_init_key_no_error():
        """
        Pass through all songs in the dataset, make sure no errors raised during training
        """
        from musicnlp.preprocess import get_dataset, KeySampleDataset

        # dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, ' \
        #           'meta={mode=melody, prec=5, th=1}, 2022-04-10_12-51-01'
        # dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
        #           'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-10_19-49-52'
        dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, ' \
                  'meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
        dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, n=10269, ' \
                  'meta={mode=melody, prec=5, th=1}, 2022-04-17_11-52-15'
        dnms = [dnm_909, dnm_lmd]

        n_sample = None
        tokenizer = MusicTokenizer(precision=5, model_max_length=2048)  # TODO: hard-code for now
        ic(tokenizer)
        # dset = get_dataset(
        #     dataset_names=dnms, map_func=lambda x: tokenizer(
        #         x['score'], padding='max_length', truncation=True),
        #     remove_columns=['title', 'score', 'duration'], n_sample=n_sample, shuffle_seed=seed
        # )
        dset = KeySampleDataset.from_hf(dnms, tokenizer=tokenizer, get_dataset_kwargs=dict(n_sample=n_sample))
        # effectively get the fist tokens of model size, simulating training data-loading
        for split, ds in dset.items():
            strt, end = 4900, len(ds)
            for i in tqdm(range(strt, end), desc=split, unit='sample'):
                d = ds[i]
                # ic(d)
                # text = tokenizer.decode(d['input_ids'])
                # ic(text)
                # im.get_init_key_est(text)
                ids = np.array(d['input_ids']).reshape(1, -1)  # dummy batch dim
                # ic(ids.shape)
                # effectively we're only checking the ground-truth init key part
                im(ids, ids)
                # exit(1)
    # check_init_key_no_error()
    profile_runtime(check_init_key_no_error)
