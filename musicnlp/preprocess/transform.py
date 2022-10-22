"""
Augmentations to a song, including
    1) inserting key, and hence
    2) shift scale degree ordinal for each pitch w.r.t. a key
    3) Mixup the relative order of melody & bass
"""

__all__ = ['KeyInsert', 'PitchShift', 'ChannelMixer']


import random
from typing import List, Dict, Union

from stefutil import *
from musicnlp.vocab import VocabType, ElmType, Channel, MusicElement, MusicVocabulary, MusicTokenizer
from musicnlp.preprocess.key_finder import ScaleDegreeFinder
from musicnlp.preprocess.music_converter import MusicConverter


class Transform:
    def __init__(self, return_as_list: bool = False):
        self.return_as_list = return_as_list

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        raise NotImplementedError


class KeyInsert(Transform):
    def __init__(self, vocab: MusicVocabulary = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab

    def __call__(self, text: str, key: Union[str, Dict[str, float]]):
        toks = text if isinstance(text, list) else text.split()
        assert self.vocab.type(toks[0]) == VocabType.time_sig  # sanity check data well-formed
        assert self.vocab.type(toks[1]) == VocabType.tempo

        if isinstance(key, dict):
            key = self.vocab(pt_sample(key))[0]
        toks.insert(2, key)
        return toks if self.return_as_list else ' '.join(toks)


class PitchShift(Transform):
    def __init__(self, vocab_step: MusicVocabulary = None, vocab_degree: MusicVocabulary = None, **kwargs):
        super().__init__(**kwargs)

        self.sdf = ScaleDegreeFinder()
        self.vocab_step = PitchShift._load_vocab(vocab=vocab_step, target_kind='step')
        self.vocab_degree = PitchShift._load_vocab(vocab=vocab_degree, target_kind='degree')

    @staticmethod
    def _load_vocab(vocab: MusicVocabulary = None, target_kind: str = None):
        if vocab is None:
            vocab = MusicVocabulary(pitch_kind=target_kind)
        else:
            assert vocab.pitch_kind == target_kind
        return vocab

    def __call__(self, text: Union[str, List[str]]):
        toks = text if isinstance(text, list) else text.split()
        key = toks[2]
        assert self.vocab_step.type(key) == VocabType.key  # sanity check; doesn't matter which vocab

        def _shift_single(tok: str) -> str:  # Convert from `step` to `degree` pitch
            # doesn't matter which vocab
            if self.vocab_step.type(tok) == VocabType.pitch and tok != self.vocab_step.rest:
                step = self.vocab_step.get_pitch_step(tok)
                deg = self.sdf.map_single(note=step, key=key)
                midi = self.vocab_step.compact(tok)   # doesn't matter which vocab
                return self.vocab_degree.note2pitch_str(note=midi, degree=deg)
            else:
                return tok
        toks = [_shift_single(tok) for tok in toks]

        sanity_check = True
        if sanity_check:
            ori = ' '.join(text)
            new = ' '.join(toks)
            mic(ori[:200], new[:200])
        return toks if self.return_as_list else ' '.join(toks)


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
        self.mc = MusicConverter(mode='full', precision=precision, vocab=vocab)
        self.vocab = self.mc.vocab

        ca(channel_mixup=mode)
        self.mode = mode

    def __call__(self, text: Union[str, List[str]]) -> str:
        out = self.mc.str2notes(text, group=True)

        sanity_check = True
        # sanity_check = False
        if sanity_check:
            for elms in out.elms_by_bar:
                mixed = self._mix_up_bar_notes(elms)
                d = self.mc.split_notes(mixed)
                mic(elms, mixed, d)
                recon = [
                    MusicElement(type=ElmType.melody), *d['melody'], MusicElement(type=ElmType.bass), *d['bass']
                ]
                assert elms == recon  # sanity check reconstruction, no info loss
            exit(1)
        ret = self.vocab.music_elm2toks(out.time_sig) + self.vocab.music_elm2toks(out.tempo)
        if out.key:
            ret += self.vocab.music_elm2toks(out.key)
        ret += sum((self._bar_music_elms2str(elms) for elms in out.elms_by_bar), start=[])
        ret += [self.vocab.end_of_song]
        return ret if self.return_as_list else ' '.join(ret)

    def _bar_music_elms2str(self, elms: List[MusicElement]):
        return [self.vocab.start_of_bar] + sum(
            (self.vocab.music_elm2toks(e) for e in self._mix_up_bar_notes(elms)), start=[]
        )

    @staticmethod
    def _bin_sample() -> bool:
        return random.randint(0, 1) == 0

    def _mix_up_bar_notes(self, elms: List[MusicElement]) -> List[MusicElement]:
        d_notes = self.mc.split_notes(elms)
        notes_m, notes_b = d_notes['melody'], d_notes['bass']

        if self.mode == 'full':
            notes_m, notes_b = iter(notes_m), iter(notes_b)
            elms = []
            note_m, note_b = next(notes_m, None), next(notes_b, None)
            c_cur, c_prev = None, None
            add_mel = None
            while note_m and note_b:
                add_mel = ChannelMixer._bin_sample()
                c_cur = Channel.melody if add_mel else Channel.bass
                diff_c = c_cur != c_prev
                if diff_c:
                    elms.append(ChannelMixer.e_m if add_mel else ChannelMixer.e_b)
                if c_cur == Channel.melody:
                    elms.append(note_m)
                    note_m = next(notes_m, None)
                else:
                    elms.append(note_b)
                    note_b = next(notes_b, None)
                c_prev = c_cur
            assert add_mel is not None  # sanity check
            if note_m:
                if not add_mel:
                    elms.append(ChannelMixer.e_m)
                elms.append(note_m)
                elms.extend(notes_m)
            else:
                if add_mel:
                    elms.append(ChannelMixer.e_b)
                assert note_b
                elms.append(note_b)
                elms.extend(notes_b)
            return elms
        else:  # `swap`
            if ChannelMixer._bin_sample():
                return elms
            else:  # swap at 50% chance
                return [ChannelMixer.e_b, *notes_b, ChannelMixer.e_m, *notes_m]
