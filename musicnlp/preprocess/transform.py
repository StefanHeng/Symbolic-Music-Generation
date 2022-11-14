"""
Augmentations to a song, including
    1) inserting key, and hence
    2) shift scale degree ordinal for each pitch w.r.t. a key
    3) Mixup the relative order of melody & bass
"""

__all__ = [
    'SanitizeRare', 'RandomCrop',
    'KeyInsert', 'TokenPitchShift', 'PitchShift', 'AugmentKey', 'CombineKeys', 'ToMidiPitch',
    'ChannelMixer'
]


from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass

import torch

from stefutil import *
from musicnlp.vocab import (
    Song, VocabType, ElmType, Channel, MusicElement, MusicVocabulary, nrp, MusicTokenizer, key_ordinal2str
)
from musicnlp.preprocess.key_finder import ScaleDegreeFinder
from musicnlp.preprocess.music_converter import MusicConverter, MusicElm


class Transform:
    def __init__(self, return_as_list: bool = False):
        self.return_as_list = return_as_list

    def __call__(self, text: Song) -> Song:
        raise NotImplementedError


class SanitizeRare(Transform):
    # See `MusicVocabulary.sanitize_rare_tokens`
    def __init__(self, vocab: MusicVocabulary = None, for_midi: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab

        # self.for_midi = for_midi  # preserve the rare pitches as they can be converted to midi pitch later
        self.for_midi = for_midi  # see `MusicVocabulary.sanitize_rare_tokens`

    def __call__(self, text: Song) -> Song:
        toks = text if isinstance(text, list) else text.split()
        # for tok in toks:
        #     if self.vocab.sanitize_rare_token(tok, for_midi=self.for_midi) == self.vocab.rare_pitch:
        #         mic(tok)
        #         mic(' '.join(toks))
        #         raise NotImplementedError
        toks = [self.vocab.sanitize_rare_token(tok, for_midi=self.for_midi) for tok in toks]
        # toks = [self.vocab.sanitize_rare_token(tok) for tok in toks]
        return toks if self.return_as_list else ' '.join(toks)


class RandomCrop(Transform):
    """
    Crop segments of the song for training

    Since song sequences are typically longer than model max length, try to make use of them all
    """
    def __init__(self, vocab: MusicVocabulary = None, min_seg_length: int = 8, crop_mult: int = 1, **kwargs):
        """
        :param start_of_bar: token for start of bar
        :param min_seg_length: minimum length of a cropped song segment
        :param crop_mult: Distance between 2 consecutive crop points, in unit of bar
            TODO: higher song fidelity to crop at multiple of 4 bars?
        """
        super().__init__(**kwargs)
        self.vocab = vocab
        self.min_seg_length = min_seg_length
        self.crop_mult = crop_mult

    def __call__(self, text: Song) -> Song:
        toks = text if isinstance(text, list) else text.split()
        idxs_bar = [i for i, tok in enumerate(toks) if tok == self.vocab.start_of_bar]
        n_bar = len(idxs_bar)
        if n_bar > self.min_seg_length:
            high = len(idxs_bar) - self.min_seg_length

            idx = 0
            if self.crop_mult == 1:
                idx = torch.randint(low=0, high=high+1, size=(1,)).item()
            else:
                if high >= self.crop_mult:
                    high = high // self.crop_mult
                    idx = torch.randint(low=0, high=high+1, size=(1,)).item() * self.crop_mult
            if idx != 0:
                toks = self._crop(toks=toks, idx=idx, idxs_bar=idxs_bar)

            sanity_check = False
            # sanity_check = True
            if sanity_check:
                n_bar_ = sum([tok == self.vocab.start_of_bar for tok in toks])
                assert n_bar_ >= self.min_seg_length
                assert n_bar - n_bar_ == idx
                _high = high if self.crop_mult == 1 else high * self.crop_mult
                mic(n_bar, _high, idx)
                if _high == idx:  # this should happen
                    raise NotImplementedError
        return toks if self.return_as_list else ' '.join(toks)

    def _crop(self, toks: List[str] = None, idx: int = None, idxs_bar: List[int] = None) -> List[str]:
        global_toks = toks[:idxs_bar[0]]
        bar_list = [toks[idx:idxs_bar[i + 1]] for i, idx in enumerate(idxs_bar[:-1])] + [toks[idxs_bar[-1]:]]

        sanity_check = False
        # sanity_check = True
        if sanity_check:
            assert global_toks + sum(bar_list, start=[]) == toks
        return global_toks + [self.vocab.omitted_segment] + list(chain_its(bar_list[idx:]))  # faster


class KeyInsert(Transform):
    def __init__(self, vocab: MusicVocabulary = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab

    def __call__(self, text: Song, key: Union[str, Dict[str, float]]) -> Song:
        toks = text if isinstance(text, list) else text.split()
        assert self.vocab.type(toks[0]) == VocabType.time_sig  # sanity check data well-formed
        assert self.vocab.type(toks[1]) == VocabType.tempo

        if isinstance(key, dict):
            key = pt_sample(key)
        toks.insert(2, self.vocab(key)[0])
        return toks if self.return_as_list else ' '.join(toks)


class TokenPitchShift:
    """
    Convert from `step` pitch to `degree` pitch, see `PitchShift`
    """
    sdf = ScaleDegreeFinder()

    def __init__(self, vocab_step: MusicVocabulary = None, vocab_degree: MusicVocabulary = None, key_token: str = None):
        self.vocab_step = vocab_step
        self.vocab_degree = vocab_degree

        self.key_meta = None
        self._key_token = key_token

    @property
    def key_token(self) -> str:
        return self._key_token

    @key_token.setter
    def key_token(self, val: str):
        if val != self._key_token:
            self._key_token = val
            self.key_meta = self.vocab_step.tok2meta(self._key_token)

    def __call__(self, tok: str) -> str:
        # doesn't matter which vocab
        if nrp(tok):  # TODO: process rare step tok tokens as many edge cases? but not much of them anyway...
            assert tok in self.vocab_step  # expect all rare pitch tokens sanitized for correct behavior
            step = self.vocab_step.get_pitch_step(tok)
            deg = TokenPitchShift.sdf.map_single(note=step, key=self.key_meta)
            midi = self.vocab_step.pitch_tok2midi_pitch_meta(tok)

            # Edge case, rare pitch token that's part of vocab, see `MusicVocabulary::_get_all_unique_pitches`
            if midi == -12:
                assert tok == 'p_1/-2_B'
                midi += 12
                # TODO: debugging
                assert self.vocab_degree.meta2tok(kind=VocabType.pitch, meta=(midi, deg)).startswith('p_1/-1_')
            elif midi == 131:
                assert tok == 'p_12/9_C'
                midi -= 12
                assert self.vocab_degree.meta2tok(kind=VocabType.pitch, meta=(midi, deg)).startswith('p_12/8_')
            return self.vocab_degree.meta2tok(kind=VocabType.pitch, meta=(midi, deg))
        else:
            return tok


class PitchShift(Transform):
    def __init__(self, vocab_step: MusicVocabulary = None, vocab_degree: MusicVocabulary = None, **kwargs):
        super().__init__(**kwargs)

        self.sdf = ScaleDegreeFinder()
        self.vocab_step = PitchShift._load_vocab(vocab=vocab_step, target_kind='step')
        self.vocab_degree = PitchShift._load_vocab(vocab=vocab_degree, target_kind='degree')

        self.tps = TokenPitchShift(vocab_step=self.vocab_step, vocab_degree=self.vocab_degree)

    @staticmethod
    def _load_vocab(vocab: MusicVocabulary = None, target_kind: str = None):
        if vocab is None:
            vocab = MusicVocabulary(pitch_kind=target_kind)
        else:
            assert vocab.pitch_kind == target_kind
        return vocab

    def __call__(self, text: Song) -> Song:
        toks = text if isinstance(text, list) else text.split()
        key = toks[2]
        assert self.vocab_step.type(key) == VocabType.key  # sanity check; doesn't matter which vocab

        self.tps.key_token = key
        toks = [self.tps(tok) for tok in toks]

        sanity_check = False
        # sanity_check = True
        if sanity_check:
            ori = ' '.join(text) if isinstance(text, list) else text
            new = ' '.join(toks)
            mic(ori[:100])
            mic(new[:100])

            # new = ' '.join(self.vocab_degree.colorize_token(tok) for tok in toks)
            # print(f'new: {new[:400]}')
            raise NotImplementedError
        return toks if self.return_as_list else ' '.join(toks)


class AugmentKey:
    def __init__(self, vocab: MusicVocabulary, return_as_list: bool = False):
        if vocab:
            assert vocab.pitch_kind == 'degree'
            self.vocab = vocab
        else:
            self.vocab = MusicVocabulary(pitch_kind='degree')

        self.ki = KeyInsert(vocab=self.vocab, return_as_list=True)
        self.ps = PitchShift(vocab_degree=self.vocab, return_as_list=return_as_list)

    def __call__(self, pair: Tuple[str, str]) -> Song:
        txt, key = pair
        txt = self.ki(text=txt, key=key)
        return self.ps(text=txt)


class CombineKeys:
    """
    Class instead of local function for pickling

    Map for vanilla training where keys need to be separately passed
    """
    n_key = len(key_ordinal2str)

    def __init__(self, tokenizer: MusicTokenizer = None):
        self.tokenizer = tokenizer

        self.sr = SanitizeRare(vocab=tokenizer.vocab)

    def __call__(self, sample):
        txt = self.sr(sample['score'])
        ret = self.tokenizer(txt, padding='max_length', truncation=True)
        keys: List[Dict[str, Optional[float]]] = sample['keys']
        # convert to a tensor format to eventually pass down to `compute_loss` and `compute_metrics`
        # -1 for metric computation to ignore
        ret['key_scores'] = [[(d[key_ordinal2str[i]] or -1) for i in range(CombineKeys.n_key)] for d in keys]
        return ret


class ToMidiPitch(Transform):
    """
    Convert songs with music-theory annotated pitch (pitch kind `step`, `degree`) to midi pitch
        Intended for rendering output
    """
    def __init__(self, vocab: MusicVocabulary = None, **kwargs):
        super().__init__(**kwargs)
        assert vocab.pitch_kind != 'midi'
        self.vocab = vocab or MusicVocabulary(pitch_kind='step')

    def __call__(self, text: Song) -> Song:
        toks = text if isinstance(text, list) else text.split()
        _v = MusicVocabulary(pitch_kind='midi')
        for tok in toks:
            if nrp(tok):
                try:
                    assert self.vocab.pitch_tok2midi_pitch_tok(tok) in _v
                except Exception as e:
                    mic(tok)
                    raise e
        toks = [(self.vocab.pitch_tok2midi_pitch_tok(tok) if nrp(tok) else tok) for tok in toks]

        sanity_check = False
        if sanity_check:
            ori = text[:400]
            new = ' '.join(toks)[:400]
            mic(ori, new)
            raise NotImplementedError
        return toks if self.return_as_list else ' '.join(toks)


@dataclass
class BarChannelSplitOutput:
    melody: List[MusicElm] = None
    bass: List[MusicElm] = None


class ChannelMixer(Transform):
    """
    Reorder notes across channels while keeping the order within the channel

    For each change of channel, prefix it
    """
    e_m, e_b = MusicElement(type=ElmType.melody), MusicElement(type=ElmType.bass)

    def __init__(
            self, precision: int = 5, vocab: MusicVocabulary = None, mode: str = 'full', **kwargs
    ):
        super().__init__(**kwargs)
        self.mc = MusicConverter(mode='full', precision=precision, vocab_midi=vocab)
        self.vocab = self.mc.vocabs.midi  # pitch kind doesn't matter

        ca(channel_mixup=mode)
        self.mix_mode = mode

    @staticmethod
    def _bin_sample() -> bool:
        return torch.randint(2, (1,)).item() == 0

    def __call__(self, text: Song) -> Song:
        out = self.mc.str2tok_elms(text)
        toks = [out.time_sig, out.tempo]
        if out.key:
            toks.append(out.key)
        if out.omit:
            toks.append(out.omit)

        toks += list(chain_its((self._mix_up_bar_toks(elms) for elms in out.elms_by_bar)))  # Faster
        toks.append(self.vocab.end_of_song)

        # sanity_check = True
        sanity_check = False
        if sanity_check:  # Should be able to re-construct the text w/ default ordering
            _text = ' '.join(text)
            mic(_text)
            ori_out = self.mc.str2music_elms(' '.join(toks), group=True)
            ori_toks = self.vocab.music_elm2toks(ori_out.time_sig) + self.vocab.music_elm2toks(ori_out.tempo)
            if ori_out.key:
                ori_toks += self.vocab.music_elm2toks(ori_out.key)
            for bar in ori_out.elms_by_bar:
                d_notes = self.mc.split_notes(bar)
                notes = [ChannelMixer.e_m, *d_notes['melody'], ChannelMixer.e_b, *d_notes['bass']]
                ori_toks += self._bar_music_elms2str(notes, mix=False)
            # ori_toks += [self.vocab.end_of_song]
            ori = ' '.join(ori_toks)

            mic(ori[:200], _text[:200])
            mic(ori == _text)
            raise NotImplementedError
        return toks if self.return_as_list else ' '.join(toks)

    def _split_bar_toks(self, elms: List[MusicElm]):
        melody, bass = [], []
        it = iter(elms)

        e1 = next(it)
        assert e1[0] in [self.vocab.start_of_melody, self.vocab.start_of_bass]
        c = Channel.melody if e1[0] == self.vocab.start_of_melody else Channel.bass  # 1st note have to specify this

        for e in it:
            if e[0] == self.vocab.start_of_melody:
                c = Channel.melody
            elif e[0] == self.vocab.start_of_bass:
                c = Channel.bass
            else:
                (melody if c == Channel.melody else bass).append(e)
        return BarChannelSplitOutput(melody=melody, bass=bass)

    def _mix_up_bar_toks(self, elms: List[MusicElm]) -> List[str]:
        out = self._split_bar_toks(elms)
        elms_m, elms_b = out.melody, out.bass

        if self.mix_mode == 'full':
            elms_m, elms_b = iter(elms_m), iter(elms_b)
            ret = []

            elm_m, elm_b = next(elms_m, None), next(elms_b, None)
            curr, prev = None, None
            add_to_melody = None
            while elm_m and elm_b:
                add_to_melody = self._bin_sample()
                curr = [self.vocab.start_of_melody] if add_to_melody else [self.vocab.start_of_bass]
                diff_channel = curr != prev
                if diff_channel:
                    ret += curr
                if add_to_melody:
                    ret += elm_m
                    elm_m = next(elms_m, None)
                else:
                    ret += elm_b
                    elm_b = next(elms_b, None)
                prev = curr
            if elm_m:
                if not add_to_melody:
                    ret += [self.vocab.start_of_melody]
                ret += elm_m
                for elm_m in elms_m:
                    ret += elm_m
            else:
                assert elm_b
                if add_to_melody:
                    ret += [self.vocab.start_of_bass]
                ret += elm_b
                for elm_b in elms_b:
                    ret += elm_b
        else:  # `swap`
            toks_m = [self.vocab.start_of_melody] + sum(elms_m, start=[])
            toks_b = [self.vocab.start_of_bass] + sum(elms_b, start=[])
            ret = (toks_m + toks_b) if self._bin_sample() else (toks_b + toks_m)
        return [self.vocab.start_of_bar] + ret

    def __call__obsolete(self, text: Song) -> Song:
        # Going through the `str => MusicElement => str` conversion is slow
        out = self.mc.str2music_elms(text, group=True)

        # sanity_check = True
        sanity_check = False
        if sanity_check:
            for elms in out.elms_by_bar:
                mixed = self._mix_up_bar_notes(elms)
                d = self.mc.split_notes(mixed)
                mic(elms, mixed, d)
                recon = [
                    MusicElement(type=ElmType.melody), *d['melody'], MusicElement(type=ElmType.bass), *d['bass']
                ]
                assert elms == recon  # sanity check reconstruction, no info loss
            raise NotImplementedError
        toks = self.vocab.music_elm2toks(out.time_sig) + self.vocab.music_elm2toks(out.tempo)
        if out.key:
            toks += self.vocab.music_elm2toks(out.key)
        toks += sum((self._bar_music_elms2str(elms) for elms in out.elms_by_bar), start=[])
        toks += [self.vocab.end_of_song]

        # sanity_check = True
        sanity_check = False
        if sanity_check:  # Should be able to re-construct the text w/ default ordering
            _text = ' '.join(text)
            # mic('Channel Mix san check', _text)
            # mic(self.mc.vocab.pitch_kind)
            ori_out = self.mc.str2music_elms(' '.join(toks), group=True)
            ori_toks = self.vocab.music_elm2toks(ori_out.time_sig) + self.vocab.music_elm2toks(ori_out.tempo)
            if ori_out.key:
                ori_toks += self.vocab.music_elm2toks(ori_out.key)
            mic(ori_toks, ori_out.key)
            for bar in ori_out.elms_by_bar:
                d_notes = self.mc.split_notes(bar)
                notes = [ChannelMixer.e_m, *d_notes['melody'], ChannelMixer.e_b, *d_notes['bass']]
                ori_toks += self._bar_music_elms2str(notes, mix=False)
            ori_toks += [self.vocab.end_of_song]
            ori = ' '.join(ori_toks)
            mic(_text[:200], ori[:200])
            mic(ori == _text)
            raise NotImplementedError
        return toks if self.return_as_list else ' '.join(toks)

    def _bar_music_elms2str(self, elms: List[MusicElement], mix: bool = True):
        if mix:
            elms = self._mix_up_bar_notes(elms)
        return [self.vocab.start_of_bar] + sum(
            (self.vocab.music_elm2toks(e) for e in elms), start=[]
        )

    def _mix_up_bar_notes(self, elms: List[MusicElement]) -> List[MusicElement]:
        d_notes = self.mc.split_notes(elms)
        notes_m, notes_b = d_notes['melody'], d_notes['bass']

        if self.mix_mode == 'full':
            notes_m, notes_b = iter(notes_m), iter(notes_b)
            ret = []
            note_m, note_b = next(notes_m, None), next(notes_b, None)
            c_cur, c_prev = None, None
            add_mel = None
            while note_m and note_b:
                add_mel = ChannelMixer._bin_sample()
                c_cur = Channel.melody if add_mel else Channel.bass
                diff_c = c_cur != c_prev
                if diff_c:
                    ret.append(ChannelMixer.e_m if add_mel else ChannelMixer.e_b)
                if add_mel:
                    ret.append(note_m)
                    note_m = next(notes_m, None)
                else:
                    ret.append(note_b)
                    note_b = next(notes_b, None)
                c_prev = c_cur
            assert add_mel is not None  # sanity check
            if note_m:
                if not add_mel:
                    ret.append(ChannelMixer.e_m)
                ret.append(note_m)
                ret.extend(notes_m)
            else:
                if add_mel:
                    ret.append(ChannelMixer.e_b)
                assert note_b
                ret.append(note_b)
                ret.extend(notes_b)
            return ret
        else:  # `swap`; Seems to break training, pbb cos don't know which channel to start
            if ChannelMixer._bin_sample():
                return elms
            else:  # swap at 50% chance
                return [ChannelMixer.e_b, *notes_b, ChannelMixer.e_m, *notes_m]


if __name__ == '__main__':
    from tqdm.auto import tqdm

    from musicnlp.preprocess import dataset

    mic.output_width = 128

    pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')

    def profile_tsf():
        aug_key = False

        vocab = MusicVocabulary(pitch_kind='degree' if aug_key else 'step')

        if aug_key:
            songs = dataset.load_songs(pop, as_dict=False)
            out = dataset.iter_songs_n_key(songs)
            it, n = out.generator, out.total
        else:
            it = dataset.load_songs(pop)
            n = len(it)

        if aug_key:
            fn = AugmentKey(vocab=vocab)
            it = (fn(pair) for pair in it)
        else:
            fn = ChannelMixer(vocab=vocab)
            it = (fn(txt) for txt in it)
        for _ in tqdm(it, total=n):
            pass
    # profile_runtime(profile_tsf)

    def check_step_pitch_mappable_to_degree_pitch():
        vocab_step = MusicVocabulary(pitch_kind='step')
        vocab_degree = MusicVocabulary(pitch_kind='degree')

        # t = vocab_step.type('p_1/-1_C')
        # mic(t, VocabType.pitch, t == VocabType.pitch)
        # exit(1)

        mic(vocab_step.toks['pitch'])
        mic(vocab_degree.toks['pitch'])

        key_toks = vocab_step.toks['key']
        mic(len(key_toks))
        for key in key_toks:
            mic(key)
            tps = TokenPitchShift(vocab_step=vocab_step, vocab_degree=vocab_degree, key_token=key)
            for pch in vocab_step.toks['pitch']:
                tok_deg = tps(tok=pch)
                if tok_deg not in vocab_degree:
                    mic(key, pch, tok_deg)
                    raise NotImplementedError
    # check_step_pitch_mappable_to_degree_pitch()

    def check_all_degree_pitches_covered():
        # mic(_get_unique_step_pitch_midis())
        # exit(1)
        vocab = MusicVocabulary(pitch_kind='degree')

        # dnms = [pop]
        # dnms = [pop, mst]
        dnms = [pop, mst, lmd]
        songs = dataset.load_songs(*dnms)
        # for s in songs:
        #     mic(s)

        sr = SanitizeRare(vocab=MusicVocabulary(pitch_kind='step'), return_as_list=True)
        ak = AugmentKey(vocab=vocab, return_as_list=True)

        out = dataset.iter_songs_n_key(songs)
        it, n = out.generator, out.total

        for txt, key in tqdm(it, desc='Checking toks in degree vocab', total=n):
            # mic(txt, key)
            text = sr(txt)
            text = ak((text, key))
            for tok in text:
                if tok not in vocab:
                    mic(tok)
                    raise NotImplementedError(pl.i(tok))
    # check_all_degree_pitches_covered()

    def viz_transform_output():
        import musicnlp.util.music as music_util
        from musicnlp.preprocess import MusicExtractor

        fnm = 'Canon piano'
        fnm = music_util.get_my_example_songs(fnm, fmt='MXL')
        mic(fnm)

        me = MusicExtractor(mode='full', with_pitch_step=True)
        out = me(fnm, exp='str_join', return_meta=True, return_key=True)
        text, keys = out.score, out.keys

        vocab = MusicVocabulary(pitch_kind='degree')
        mc = MusicConverter(mode='full', vocab_midi=vocab)

        print(f'Extracted: {mc.visualize_str(text)}')
        mic(keys)
        key = max(keys, key=keys.get)
        mic(key)

        ak = AugmentKey(vocab=vocab)
        text = ak((text, key))
        print(f'Added key & scale degree: {mc.visualize_str(text)}')
    viz_transform_output()
