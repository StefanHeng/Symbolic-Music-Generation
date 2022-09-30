import itertools
from typing import Dict, Iterable
from fractions import Fraction
from collections import Counter

from musicnlp.util.music_lib import Dur
from musicnlp.vocab import ElmType, MusicElement, MusicVocabulary
from musicnlp.preprocess.music_converter import MusicConverter


class MusicStats:
    def __init__(self, prec: int = 5, converter_kw: Dict = None):
        self.prec = prec
        if converter_kw is None:
            converter_kw = dict()
        self.converter = MusicConverter(precision=prec, **converter_kw)
        self.vocab: MusicVocabulary = self.converter.vocab

    def vocab_type_counts(self, toks: Iterable[str], strict: bool = True) -> Dict[str, Counter]:
        """
        :param toks: Iterable of token strings
        :param strict: See `MusicVocabulary`
        :return: Counter on compact representation by the compact token types
        """
        toks = sorted(toks)
        type2toks = {k: list(v) for k, v in itertools.groupby(toks, key=lambda tok: self.vocab.type(tok))}
        type2toks = {
            k: list(self.vocab.compact(t, strict=strict) for t in v) for k, v in type2toks.items() if k in self.vocab.compacts
        }
        return {k.name: Counter(v) for k, v in type2toks.items()}

    def weighted_pitch_counts(self, toks: Iterable[str]) -> Dict[str, Dur]:
        """
        :return: counts for pitch, weighed by the duration in quarter length
        """
        if not isinstance(toks, list):
            toks = list(toks)
        notes = [elm for elm in self.converter.str2notes(toks).elms if elm.type in [ElmType.note, ElmType.tuplets]]

        def elm2notes(elm: MusicElement):
            typ, compacts = elm.type, elm.meta
            if typ == ElmType.note:
                return [compacts]
            else:
                comps_p, comp_d = compacts
                comp_d = Fraction(comp_d, len(comps_p))
                return [(comp_p, comp_d) for comp_p in comps_p]

        note_n_dur = sorted(sum((elm2notes(elm) for elm in notes), start=[]))
        pch2dur = {
            pch: [pair[1] for pair in pairs] for pch, pairs in itertools.groupby(note_n_dur, key=lambda pair: pair[0])
        }
        return {pch: sum(durs) for pch, durs in pch2dur.items()}


if __name__ == '__main__':
    import musicnlp.util.music as music_util

    ms = MusicStats()
    text = music_util.get_extracted_song_eg()
    toks_ = text.split()

    # mic(ms.vocab_type_counts(toks))
    mic(ms.weighted_pitch_counts(toks_))
