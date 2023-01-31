import itertools
from typing import Dict, Iterable
from fractions import Fraction
from collections import Counter

from stefutil import *
from musicnlp.util.music_lib import Dur
from musicnlp.vocab import ElmType, MusicElement, MusicVocabulary
from musicnlp.preprocess.music_converter import MusicConverter


class MusicStats:
    def __init__(self, prec: int = 5, converter_kw: Dict = None, pitch_kind: str = 'midi'):
        self.prec = prec
        if converter_kw is None:
            converter_kw = dict()
        self.converter = MusicConverter(precision=prec, **converter_kw)
        self.pitch_kind = pitch_kind
        self.vocab: MusicVocabulary = self.converter.pk2v[pitch_kind]

    def vocab_type_counts(self, toks: Iterable[str], strict: bool = True) -> Dict[str, Counter]:
        """
        :param toks: Iterable of token strings
        :param strict: See `MusicVocabulary`
        :return: Counter on compact representation by the compact token types
        """
        toks = sorted(toks)
        type2toks = {k: list(v) for k, v in itertools.groupby(toks, key=lambda tok: self.vocab.type(tok))}
        type2toks = {
            k: list(self.vocab.tok2meta(t, strict=strict) for t in v)
            for k, v in type2toks.items() if k in self.vocab.types_with_meta
        }
        return {k.name: Counter(v) for k, v in type2toks.items()}

    def weighted_pitch_counts(self, toks: Iterable[str]) -> Dict[str, Dur]:
        """
        :return: counts for pitch, weighed by the duration in quarter length
        """
        if not isinstance(toks, list):
            toks = list(toks)
        notes = [
            elm for elm in self.converter.str2music_elms(toks, pitch_kind=self.pitch_kind).elms
            if elm.type in [ElmType.note, ElmType.tuplets]
        ]

        def elm2notes(elm: MusicElement):
            typ, meta = elm.type, elm.meta
            if typ == ElmType.note:
                m_p, m_d = meta
                if self.pitch_kind != 'midi' and m_p != self.vocab.rare_pitch_meta:
                    m_p = m_p[0]  # remove step
                return [(m_p, m_d)]
            else:
                ms_p, m_d = meta
                if self.pitch_kind != 'midi':
                    ms_p = [(m_p if self.vocab.rare_pitch_meta == m_p else m_p[0]) for m_p in ms_p]
                m_d = Fraction(m_d, len(ms_p))
                return [(m_p, m_d) for m_p in ms_p]

        # for elm in notes:
        #     if elm2notes(elm) is None:
        #         mic(elm, elm2notes(elm))
        #         raise NotImplementedError
        pch_n_dur = sum((elm2notes(elm) for elm in notes), start=[])
        # Filter out rare pitch & durations
        pch_n_dur = [
            (p, d) for p, d in pch_n_dur if p != self.vocab.rare_pitch_meta and d != self.vocab.rare_duration_meta
        ]
        try:
            pch_n_dur = sorted(pch_n_dur)
        except Exception as e:
            mic(pch_n_dur, self.vocab.rare_pitch_meta)
            raise e
        pch2dur = {
            pch: [pair[1] for pair in pairs] for pch, pairs in itertools.groupby(pch_n_dur, key=lambda pair: pair[0])
        }
        return {pch: sum(durs) for pch, durs in pch2dur.items()}


if __name__ == '__main__':
    import musicnlp.util.music as music_util

    ms = MusicStats()
    text = music_util.get_extracted_song_eg()
    toks_ = text.split()

    # mic(ms.vocab_type_counts(toks))
    mic(ms.weighted_pitch_counts(toks_))
