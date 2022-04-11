import music21
from itertools import filterfalse
from collections import Counter

from musicnlp.util import *
from musicnlp.util.train import PT_LOSS_PAD
from musicnlp.vocab import VocabType, MusicTokenizer


class IkrMetric:
    """
    Vectorized metric of matched keys per pitch, based on `_get_off_key_ratio`
    """
    def __init__(self, tokenizer: MusicTokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab

    def __call__(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """
        Arguments should be batched autoregressive transformer input & output tokens of the same shape in 2D
        """
        assert preds.shape == labels.shape, \
            f'Input and label shapes do not match, {logi(preds.shape)} vs {logi(labels.shape)}'
        # switch from HF CLM training pad id to the tokenizer pad it
        labels[labels == PT_LOSS_PAD] = self.tokenizer.pad_token_id
        gens = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        gts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return 1 - np.mean([self.get_off_key_ratio(gen, gt)for gen, gt in zip(gens, gts)])

    def get_init_key_est(self, gt_token_seq: Union[str, List[str]], num_bars: int = 4):
        tok_lst = gt_token_seq.split() if isinstance(gt_token_seq, str) else gt_token_seq

        # Heuristics to determine starting bar
        bar_idx = [idx for idx, tok in enumerate(tok_lst) if tok in [self.vocab.start_of_bar, self.tokenizer.eos_token]]
        assert len(bar_idx) > num_bars + 1, \
            f'Not enough bars for key estimation: expect at least {logi(num_bars + 1)} total bars in music, ' \
            f'got {logi(len(bar_idx))}'

        pitch_lst = list(filterfalse(lambda x: self.vocab.type(x) != VocabType.pitch, tok_lst[:bar_idx[num_bars]]))
        key_cls = [music21.pitch.Pitch(midi=self.vocab.compact(p)).pitchClass for p in pitch_lst]
        key_est = Counter(key_cls).most_common()[0][0]
        return key_est

    def get_off_key_ratio(self, gen_token_text: Union[str, List[str]], gt_token_text: Union[str, List[str]]) -> float:
        """
        For a single song

        Both arguments are either jointed string of tokens or split list of tokens
        """
        tok_lst = gen_token_text.split() if isinstance(gen_token_text, str) else gen_token_text
        target_key = self.get_init_key_est(gt_token_text)
        pitch_lst = list(filterfalse(lambda x: self.vocab.type(x) != VocabType.pitch, tok_lst))
        num_toks = len(pitch_lst)
        if num_toks == 0:  # No pitch found, assume every pitch is off-note
            return 1
        # Heuristics
        # TODO: add more music theories later
        key_thres, key_count, num_off_key = 8, 0, 0
        prev_p_cls = None
        num_off_key = 0
        for p in pitch_lst:
            p_cls = music21.pitch.Pitch(midi=self.vocab.compact(p)).pitchClass
            if p_cls != target_key:
                num_off_key += 1
                # TODO: add cross-bar detection
                if prev_p_cls is not None and p_cls == prev_p_cls:
                    key_count += 1
                    if key_count >= key_thres:
                        target_key = p_cls
                        key_count = 0
                prev_p_cls = p_cls
            else:
                prev_p_cls = None
        return num_off_key / num_toks


if __name__ == '__main__':
    from icecream import ic

    import musicnlp.util.music as music_util

    im = IkrMetric(MusicTokenizer())

    def check_key_metric():
        text = music_util.get_extracted_song_eg(k='平凡之路')  # this one has tuplets
        ic(text[:200])
        ic(im.get_off_key_ratio(text, text))
    # check_key_metric()

    def check_init_key_no_error():
        """
        Pass through all songs in the dataset, make sure no errors raised during training
        """
        from musicnlp.preprocess import get_dataset

        dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-51-01'
        dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
                  'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-10_19-49-52'
        dnms = [dnm_909, dnm_lmd]

        n_sample = None
        seed = config('random-seed')
        tokenizer = MusicTokenizer(prec=5)
        tokenizer.model_max_length = 2048  # TODO: hard-code for now
        dset = get_dataset(
            dataset_names=dnms, map_func=lambda x: tokenizer(x['score'], padding='max_length', truncation=True),
            remove_columns=['title', 'score', 'duration'], n_sample=n_sample, shuffle_seed=seed
        )
        # effectively get the fist tokens of model size, simulating training data-loading
        for split, ds in dset.items():
            ic(split)
            for d in tqdm(ds):
                # ic(d)
                # text = tokenizer.decode(d['input_ids'])
                # ic(text)
                # im.get_init_key_est(text)
                ids = np.array(d['input_ids']).reshape(1, -1)  # dummy batch dim
                # ic(ids.shape)
                im(ids, ids)  # effectively we're only checking the ground-truth init key part
                # exit(1)
    check_init_key_no_error()
