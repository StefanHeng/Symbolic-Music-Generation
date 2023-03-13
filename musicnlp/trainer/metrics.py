from typing import List, Union, Dict, Optional
from itertools import filterfalse
from collections import Counter

import numpy as np
import torch
import music21

from stefutil import *
from musicnlp.util.train import PT_LOSS_PAD
from musicnlp.vocab import VocabType, MusicTokenizer
from musicnlp.vocab.elm_type import *


MetricTensor = Union[np.ndarray, torch.Tensor]


class IkrMetric:
    """
    Vectorized metric of matched keys per pitch, based on `_get_off_key_ratio`
    """

    def __init__(
            self, tokenizer: MusicTokenizer, n_init_bars: int = 4, mode: str = 'vanilla', clm_pred_shifted: bool = False
    ):
        """
        :param tokenizer: tokenizer
        :param n_init_bars: Number of bars for heuristic key estimation
            Obsolete, see `musicnlp.preprocess.key_finder`
        :param mode: Training mode, one of ['vanilla', 'key-aug']
            If 'vanilla', compute with weighted average of all possible keys
            If 'key-aug', compute with the key passed in at 3rd token
        """
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.n_init_bars = n_init_bars

        ca.check_mismatch('Training Mode for IKR', mode, ['vanilla', 'ins-key'])
        self.mode = mode

        self.clm_pred_shifted = clm_pred_shifted

    def __call__(self, preds: MetricTensor, labels: MetricTensor, key_scores: Optional[MetricTensor] = None) -> float:
        """
        Arguments should be batched autoregressive transformer input & output tokens of the same shape in 2D
        """
        if self.clm_pred_shifted:
            labels = labels[:, 1:]
        assert preds.shape == labels.shape, \
            f'Input and label shapes do not match, {pl.i(preds.shape)} vs {pl.i(labels.shape)}'
        ikrs = []
        if self.mode == 'vanilla':
            for pred, label, key_scores_ in zip(preds, labels, key_scores):
                if isinstance(key_scores_, torch.Tensor):
                    key_scores_ = key_scores_.cpu().numpy()
                ords, scores = zip(*[(ord_key, score) for ord_key, score in enumerate(key_scores_) if score > 0])
                pred = pred[label != PT_LOSS_PAD]
                _ikrs = [self.get_in_key_ratio(pred, key_ordinal2key_enum[o]) for o in ords]
                ikrs.append(np.average(_ikrs, weights=scores))
        elif self.mode == 'ins-key':
            for pred, label in zip(preds, labels):
                key_tok_id = label[1 if self.clm_pred_shifted else 2]  # expect labels to be well-formed
                if not self.vocab.type(key_tok_id) == VocabType.key:
                    tok = self.tokenizer.decode([key_tok_id])
                    raise ValueError(f'Expect key token at 3rd position of label, got {pl.i(key_tok_id)}:{pl.i(tok)}')
                ikrs.append(self.get_in_key_ratio(pred[label != PT_LOSS_PAD], self.vocab.tok2meta(key_tok_id)))
        return np.array(ikrs).mean()

    def get_init_key_est(self, gt_token_seq: Union[str, List[str]]):
        tok_lst = gt_token_seq.split() if isinstance(gt_token_seq, str) else gt_token_seq

        # Heuristics to determine starting bar
        bar_idx = [idx for idx, tok in enumerate(tok_lst) if tok == self.vocab.start_of_bar]
        assert len(bar_idx) > self.n_init_bars + 1, \
            f'Not enough bars for key estimation: expect at least {pl.i(self.n_init_bars + 1)} total bars in music, ' \
            f'got {pl.i(len(bar_idx))}'

        pitch_lst = list(
            filterfalse(lambda x: self.vocab.type(x) != VocabType.pitch, tok_lst[:bar_idx[self.n_init_bars]])
        )
        key_cls = [music21.pitch.Pitch(
            midi=self.vocab.tok2meta(p)).pitchClass for p in pitch_lst]
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
            midi=self.vocab.tok2meta(p)).pitchClass for p in pitch_lst])
        stats_pitch_cls_str = Counter([music21.pitch.Pitch(
            midi=self.vocab.tok2meta(p)).name for p in pitch_lst])
        mic(stats_pitch_cls_str)
        mic(stats_pitch_cls_int)

    def get_in_key_ratio(self, preds: List[int], key: Key) -> float:
        # Extract midi values for all available pitches
        lst_pch = self.tokenizer.ids2pitches(preds, include_rest_pitch=False)
        num_toks = len(lst_pch)
        if num_toks == 0:  # No pitch found, assume every pitch is off-note
            return 0
        else:
            key_type, key_name = key_enum2tuple[key]  # Process the given key

            pitch_midi = np.array(lst_pch)
            key_offset = key_offset_dict[key_name]
            pred_offset = ((pitch_midi % 12) - key_offset) % 12

            in_key_ratio = sum(x not in OFFKEY_OFFSET[key_type] for x in pred_offset) / num_toks
            return in_key_ratio


if __name__ == '__main__':
    import json
    from os.path import join as os_join

    from tqdm import tqdm

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
        with open(os_join(music_util.get_processed_path(), f'{song_name}.json'), 'w') as f:
            json.dump(dict(score=score), f, indent=2)
    # write_eg_song_json(song_nm)

    def get_eg_song_key(song_name: str = 'Merry Go Round of Life'):
        fnm = music_util.get_my_example_songs(song_name, fmt='MXL')

        kf = KeyFinder(fnm)
        keys = kf.__call__(return_type="enum")
        mic(keys)
        # mic(kf.find_scale_degrees(keys))
    # get_eg_song_key(song_nm)

    def check_key_metric():
        im = IkrMetric(MusicTokenizer(), n_init_bars=2)

        # text = music_util.get_extracted_song_eg(
        # k='平凡之路')  # this one has tuplets
        with open(os_join(music_util.get_processed_path(), f'{song_nm}.json'), 'r') as f:
            text = json.load(f)['score']
        # Get keys from Key_Finder
        fnm = music_util.get_my_example_songs(song_nm, fmt='MXL')
        kf = KeyFinder(fnm)
        keys = kf.__call__(return_type="enum")
        # mic(text[:200])
        # mic(im.get_off_key_ratio(text, Key.AfMaj))
        exp_out = [im.get_in_key_ratio(text, key) for key in keys]
        mic(exp_out)
        mic(f"Average IKR for {song_nm}: {np.round(np.mean(exp_out), 5)}")
        # mic(im.get_in_key_ratio(text, Key.DMin))
    # check_key_metric()

    def check_init_key_no_error():
        """
        Pass through all songs in the dataset, make sure no errors raised during training
        """
        from musicnlp.preprocess import dataset

        pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')
        dnms = [pop, mst, lmd]

        im = IkrMetric(MusicTokenizer(), n_init_bars=2)

        n_sample = None
        tokenizer = MusicTokenizer(precision=5, model_max_length=2048)  # TODO: hard-code for now
        mic(tokenizer)
        # dset = get_dataset(
        #     dataset_names=dnms, map_func=lambda x: tokenizer(
        #         x['score'], padding='max_length', truncation=True),
        #     remove_columns=['title', 'score', 'duration'], n_sample=n_sample, shuffle_seed=seed
        # )
        dset = dataset.AugmentedDataset.from_hf(dnms, tokenizer=tokenizer, get_dataset_args=dict(n_sample=n_sample))
        # effectively get the fist tokens of model size, simulating training data-loading
        for split, ds in dset.items():
            strt, end = 4900, len(ds)
            for i in tqdm(range(strt, end), desc=split, unit='sample'):
                d = ds[i]
                # mic(d)
                # text = tokenizer.decode(d['input_ids'])
                # mic(text)
                # im.get_init_key_est(text)
                ids = np.array(d['input_ids']).reshape(1, -1)  # dummy batch dim
                # mic(ids.shape)
                # effectively we're only checking the ground-truth init key part
                im(ids, ids)
                # exit(1)
    # check_init_key_no_error()
    # profile_runtime(check_init_key_no_error)

    def check_ground_truth_ikr():
        """
        Sanity check if IKR is a good metric
            The value for actual songs should be pretty high
        """
        from musicnlp.vocab import key_str2enum
        from musicnlp.preprocess import dataset

        pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')
        dnms = [pop]
        # dnms = [pop, mst]

        tokenizer = MusicTokenizer(pitch_kind='step')
        im = IkrMetric(tokenizer=tokenizer)
        mic(tokenizer.vocab_size)

        songs: List[Dict] = dataset.load_songs(*dnms)
        it = tqdm(songs, desc='Computing IKR on ground truth', unit='song')

        reduce_kind = 'most-confident-key'  # use the IKR from the most-confident key
        mic(reduce_kind)
        # reduce_kind = 'highest-ikr'  # a cheating upper bound, when the key stays unchanged throughout the song

        ikrs = []
        for s in it:
            it.set_postfix(title=s['title'])
            scr = s['score']
            input_ids = tokenizer(scr)['input_ids']   # Tokenize the entire sequence, no padding & truncation
            ks = {key_str2enum[k]: c for k, c in s['keys'].items()}
            d_ikrs = {k: im.get_in_key_ratio(preds=input_ids, key=k) for k in ks.keys()}
            # mic(s['keys'], d_ikrs)

            if reduce_kind == 'most-confident-key':  # ~0.95 for POP909
                k_conf = max(ks, key=ks.get)
                # Does highest-confidence key always have highest IKR value? --- No
                # assert d_ikrs[k_conf] == max(d_ikrs.values())  # for ties
                ikrs.append(d_ikrs[k_conf])
            elif reduce_kind == 'highest-ikr':  # ~0.97 for POP909
                ikrs.append(max(d_ikrs.values()))
            # raise NotImplementedError
        mic(np.mean(ikrs))
    check_ground_truth_ikr()
