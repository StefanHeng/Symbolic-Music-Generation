import re
import json
import math
from os.path import join as os_join
from enum import Enum
from typing import List, Tuple, Set, Dict, Iterator, Optional, Union
from fractions import Fraction
from collections import OrderedDict

import numpy as np
import torch
import music21 as m21

from stefutil import *
from musicnlp.util.music_lib import *
import musicnlp.util.music as music_util
from musicnlp.vocab.elm_type import Key, key_str2enum


COMMON_TIME_SIGS: List[TsTup] = sorted(  # Sort first by denominator
    [(4, 4), (2, 4), (2, 2), (3, 4), (6, 8), (5, 4), (12, 8)],
    key=lambda tup_: tuple(reversed(tup_))
)
COMMON_TEMPOS: List[int] = list(range(40, 241))  # [40-240]


def is_common_time_sig(ts: Union[TimeSignature, TsTup]):
    if not hasattr(is_common_time_sig, 'COM_TS'):  # List of common time signatures
        is_common_time_sig.COM_TS = set(COMMON_TIME_SIGS)
    if isinstance(ts, TimeSignature):
        ts = (ts.numerator, ts.denominator)
    return ts in is_common_time_sig.COM_TS


def get_common_time_sig_duration_bound():
    return max([(numer/denom) for numer, denom in COMMON_TIME_SIGS]) * 4


def is_common_tempo(tempo: Union[MetronomeMark, int]):
    if not hasattr(is_common_tempo, 'COM_TEMPO'):  # List of common tempos
        is_common_tempo.COM_TEMPO = set(COMMON_TEMPOS)
    if isinstance(tempo, MetronomeMark):
        tempo = tempo.number
    return tempo in is_common_tempo.COM_TEMPO


class VocabType(Enum):
    time_sig, tempo, key, duration, pitch, special = list(range(6))

    @classmethod
    def compact(cls) -> Iterator['VocabType']:
        """
        :return: Iterator of all token types with compact representation
        """
        for i in range(5):  # `special` doesn't have a compact representation
            yield cls(i)


int_types = (int, np.int32, np.int64, torch.Tensor)
Int = Union[int, np.int32, np.int64, torch.Tensor]
Compact = Union[
    TsTup, int, Dur, Key,
    Tuple[None, None], None  # for uncommon tokens
]


class MusicVocabulary:
    """
    Stores mapping between string tokens and integer ids
    & support the conversion, from relevant `music21` objects to [`str`, `int] conversion
    """
    start_of_bar = '<bar>'
    end_of_song = '</s>'
    start_of_tuplet = '<tup>'
    end_of_tuplet = '</tup>'
    pad = '[PAD]'  # Needed for type-checking, see `musicnlp.models.metric.get_in_key_ratio`

    sep = '_'  # Separation
    time_sig_pref = 'TimeSig'
    tempo_pref = 'Tempo'
    dur_pref = 'd'
    uncommon_time_sig = f'{time_sig_pref}{sep}uncommon'
    uncommon_low_tempo = f'{tempo_pref}{sep}low'
    uncommon_high_tempo = f'{tempo_pref}{sep}high'
    uncommon_duration = f'{dur_pref}{sep}uncommon'  # has to be too long, assuming durations are quantized
    compact_uncommon_time_sig = (None, None)  # TODO: check, may break things?
    compact_low_tempo = 40-1  # See `COMMON_TEMPOS`
    compact_high_tempo = 240+1
    compact_uncommon_duration = None
    uncom_tok2compact = {
        uncommon_time_sig: compact_uncommon_time_sig,
        uncommon_low_tempo: compact_low_tempo,
        uncommon_high_tempo: compact_high_tempo,
        uncommon_duration: compact_uncommon_duration
    }

    SPEC_TOKS = dict(
        sep=sep,
        rest='r',
        prefix_pitch='p',
        prefix_duration=dur_pref,
        start_of_tuplet=start_of_tuplet,
        end_of_tuplet=end_of_tuplet,
        start_of_bar=start_of_bar,
        end_of_song=end_of_song,
        prefix_time_sig=time_sig_pref,
        prefix_tempo=tempo_pref,
        prefix_key='Key'
    )
    # # Uncommon Time Signatures in music theory, but empirically seen in MIDI data
    # # See music_visualize.py for distribution
    # UNCOM_TSS: List[TsTup] = [
    #     # (1, 4),  # seen, a lot, from POP909
    #     #
    #     # # from LMD-cleaned_subset, only a small fraction are edge cases
    #     # (3, 2), (4, 2),
    #     # (6, 4), (7, 4), (8, 4), (12, 4), (132, 4),
    #     # (1, 8), (3, 8), (4, 8), (5, 8), (7, 8), (8, 8), (9, 8), (11, 8),
    #     # (8, 16), (16, 16),
    #     # (2, 64)
    #
    #     # ~80 uncommon ones considering POP909, MAESTRO & LMD, convert to a single uncommon token
    #     (1, 1), (2, 1), (4, 1),
    #     (1, 2), (3, 2), (4, 2), (5, 2), (6, 2), (8, 2), (10, 2), (12, 2), (74, 2),
    #     (1, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4), (11, 4), (12, 4), (15, 4), (16, 4),
    #     (18, 4), (32, 4), (66, 4), (68, 4), (131, 4), (132, 4), (133, 4), (149, 4),
    #     (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (7, 8), (8, 8), (9, 8), (10, 8), (11, 8), (13, 8), (15, 8), (16, 8),
    #     (17, 8), (18, 8), (19, 8), (20, 8), (22, 8), (24, 8), (124, 8), (134, 8), (140, 8), (252, 8),
    #     (1, 16), (2, 16), (3, 16), (4, 16), (5, 16), (6, 16), (7, 16), (8, 16), (9, 16), (10, 16),
    #     (11, 16), (12, 16), (13, 16), (15, 16), (16, 16), (18, 16), (24, 16),
    #     (12, 32), (14, 32), (32, 32), (137, 32),
    #     (2, 64), (4, 64),
    #     (64, 128)
    # ]
    # UNCOM_TPS: List[int] = [
    #     # 30, 37,  # Observed from LMD-cleaned_subset
    #     # 241, 244, 245, 246, 250, 254, 255, 256, 265, 275, 276, 278, 280, 287, 293,
    #     # 305, 334, 397
    #
    #     # ~120 edge case tempos taking up embedding storage => group them into a `HIGH` and a `LOW` token
    #     6, 10, 11, 12, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    #     241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
    #     261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 278, 279, 280,
    #     282, 283, 285, 286, 287, 288, 289, 290, 291, 293, 294, 295, 297, 299, 300,
    #     302, 305, 306, 310, 318, 320, 324, 325, 327, 329, 330, 334, 338, 339, 340, 349, 350,
    #     353, 360, 366, 368, 370, 385, 395, 397, 400,
    #     403, 406, 432, 440, 450, 451, 480, 500, 616, 700, 900
    # ]
    # Observed from LMD-cleaned_subset
    # UNCOM_DURS: List[Union[int, float]] = [
    #     # 25/4, 13/2, 15/2, 8, 77/8, 12, 29/2, 16, 24  # Observed from LMD-cleaned_subset
    # ]

    RE_INT = r'[-]?\d*'  # negative sign for `octave`
    RE1 = rf'(?P<num>{RE_INT})'
    RE2 = rf'(?P<numer>{RE_INT})/(?P<denom>{RE_INT})'

    _token_type2color: Dict[VocabType, str] = {
        VocabType.time_sig: 'r',
        VocabType.tempo: 'r',
        VocabType.key: 'r',
        VocabType.duration: 'g',
        VocabType.pitch: 'b',
        VocabType.special: 'm'
    }

    # TODO: remove, original training was without key support
    def __init__(self, precision: int = 5, color: bool = False):
        """
        :param precision: See `musicnlp.preprocess.music_extractor`
        :param color: If True, string outputs are colorized
            Update individual coloring of subsequent tokens via `__getitem__`
        """
        self.precision = precision
        self.color = color

        specs = MusicVocabulary.SPEC_TOKS  # Syntactic sugar
        sep = specs['sep']
        self.cache = dict(  # Common intermediary substrings
            pref_dur=specs['prefix_duration']+sep,
            pref_pch=specs['prefix_pitch']+sep,
            pref_time_sig=specs['prefix_time_sig']+sep,
            pref_tempo=specs['prefix_tempo']+sep,
            pref_key=specs['prefix_key']+sep,
            bot=self['start_of_tuplet'],
            eot=self['end_of_tuplet']
        )
        self.rest = self.cache['rest'] = self.cache['pref_pch'] + specs['rest']

        self.type2compact_re = {
            VocabType.duration: dict(
                int=re.compile(rf'^{self.cache["pref_dur"]}{MusicVocabulary.RE1}$'),
                frac=re.compile(rf'^{self.cache["pref_dur"]}{MusicVocabulary.RE2}$'),
            ),
            VocabType.pitch: re.compile(rf'^{self.cache["pref_pch"]}{MusicVocabulary.RE2}$'),
            VocabType.time_sig: re.compile(rf'^{self.cache["pref_time_sig"]}{MusicVocabulary.RE2}$'),
            VocabType.tempo: re.compile(rf'^{self.cache["pref_tempo"]}{MusicVocabulary.RE1}$'),
            VocabType.key: re.compile(rf'^{self.cache["pref_key"]}(?P<key>.*)$'),
        }

        self.compacts: Set[VocabType] = set(VocabType.compact())

        def elm2str(elm):
            return self(elm, color=False, return_int=False)

        def rev(time_sig):
            return tuple(reversed(time_sig))  # Syntactic sugar
        tss = [elm2str(rev(ts))[0] for ts in sorted(rev(ts) for ts in COMMON_TIME_SIGS)]
        # See music_visualize.py for distribution
        tempos = [elm2str(tp)[0] for tp in COMMON_TEMPOS]
        pitches = [self.cache['rest']] + [self.note2pitch_str(Pitch(midi=i)) for i in range(128)]
        keys = [elm2str(k)[0] for k in sorted(key_str2enum.keys())]

        # TODO: with music-theory, mod-7 scale degree, vocab size would increase
        special = [specs[k] for k in ('end_of_song', 'start_of_bar', 'start_of_tuplet', 'end_of_tuplet')]
        special.append(MusicVocabulary.pad)
        self.toks: Dict[str, List[str]] = OrderedDict(dict(
            special=special,
            time_sig=[MusicVocabulary.uncommon_time_sig, *tss],
            tempo=[MusicVocabulary.uncommon_low_tempo, *tempos, MusicVocabulary.uncommon_high_tempo],
            key=keys,
            pitch=pitches,
            duration=[MusicVocabulary.uncommon_duration, *self.get_durations(exp='str')]
        ))
        self.tok2id: Dict[str, int] = {  # back-to0back index as ids
            tok: id_ for id_, tok in enumerate(chain_its(toks for toks in self.toks.values()))
        }
        self.id2tok = {v: k for k, v in self.tok2id.items()}
        assert len(self.tok2id) == len(self.id2tok)  # Sanity check: no id collision

        # cache them for efficiency
        self.id2type: Dict[int, VocabType] = {id_: self.type(tok) for id_, tok in self.id2tok.items()}
        self.id2compact: Dict[int, Compact] = {
            id_: self.compact(tok) for id_, tok in self.id2tok.items() if self.has_compact(tok)
        }

    def to_dict(self, save=False):
        d_out = dict(
            precision=self.precision,
            special_tokens={
                'start_of_bar': MusicVocabulary.start_of_bar,
                'end_of_song': MusicVocabulary.end_of_song,
                'start_of_tuplet': MusicVocabulary.start_of_tuplet,
                'end_of_tuplet': MusicVocabulary.end_of_tuplet
            },
            vocabulary=self.tok2id,
            n_vocabulary=len(self.tok2id),
        )
        if save:
            fnm = f'{self.__class__.__qualname__}, n={len(self.tok2id)}, prec={self.precision}, {now(for_path=True)}'
            path = os_join(music_util.get_processed_path(), f'{fnm}.json')
            with open(path, 'w') as f:
                json.dump(d_out, f, indent=4)
        return d_out

    def get_durations(self, bound: int = None, exp: str = 'str') -> Union[List[str], List[Dur]]:
        """
        :param bound: The upper bound for duration within a bar, in terms of `quarterLength`
            By default, support duration up to **6**
        :param exp: return type, one of ['str`, `dur`]
            If str, returns duration token
            If dur, returns `Dur`
        :return: List of durations
        """
        if bound is None:
            # Effectively support up to 6 in terms of quarter length; TODO: support for longer duration needed?
            bound = max(ts[0]/ts[1] for ts in COMMON_TIME_SIGS) * 4
            assert bound.is_integer()
        dur_slot, denom = 4 / 2 ** self.precision, 2 ** self.precision / 4
        assert denom.is_integer()
        dur_nums = list(range(math.ceil(bound / dur_slot)))
        if exp == 'str':
            durs = [self._note2dur_str((i+1) * dur_slot) for i in dur_nums]
            return durs
        else:
            assert exp == 'dur'
            denom = int(denom)
            ret = [Fraction(i+1, denom) for i in dur_nums]
            return [int(f) if f.denominator == 1 else f for f in ret]

    def __len__(self):
        return len(self.tok2id)

    def has_compact(self, tok: Union[str, int]) -> bool:
        return self.type(tok) != VocabType.special

    def type(self, tok: Union[str, Int]) -> VocabType:
        if isinstance(tok, int_types):
            return self.id2type[int(tok)]
        else:  # order by decreasing expected frequency for efficiency
            if self.cache['pref_pch'] in tok:
                return VocabType.pitch
            elif self.cache['pref_dur'] in tok:
                return VocabType.duration
            elif self.cache['pref_time_sig'] in tok:
                return VocabType.time_sig
            elif self.cache['pref_tempo'] in tok:
                return VocabType.tempo
            elif self.cache['pref_key'] in tok:
                return VocabType.key
            else:
                return VocabType.special

    @staticmethod
    def _get_group1(tok, tpl) -> int:
        return int(tpl.match(tok).group('num'))

    @staticmethod
    def _get_group2(tok, tpl) -> Tuple[int, int]:
        m = tpl.match(tok)
        return int(m.group('numer')), int(m.group('denom'))

    def compact(self, tok: Union[str, Int], for_stats: bool = False) -> Compact:
        """
        Convert tokens to the numeric format

        More compact, intended for statistics

        Raise error is special tokens passed

        :return: If time signature, returns 2-tuple of (int, int),
            If tempo, returns integer of tempo number
            If pitch, returns the pitch MIDI number
            If duration, returns the duration quarterLength

        representation remains faithful to the token for the uncommon ones
        """
        assert self.has_compact(tok), ValueError(f'{logi(tok)} does not have a compact representation')
        if isinstance(tok, int_types):
            return self.id2compact[int(tok)]
        elif tok in MusicVocabulary.uncom_tok2compact:
            return MusicVocabulary.uncom_tok2compact[tok]
        else:
            typ = self.type(tok)
            tpl = self.type2compact_re[typ]
            if typ == VocabType.duration:
                if '/' in tok:
                    numer, denom = MusicVocabulary._get_group2(tok, tpl['frac'])
                    assert math.log2(denom).is_integer()
                    # Quantized so definitely an exact float, but keep Fraction for exact additions
                    return Fraction(numer, denom)
                else:
                    return MusicVocabulary._get_group1(tok, tpl['int'])
            elif typ == VocabType.pitch:
                if tok == self.cache['rest']:
                    return -1
                else:
                    pch, octave = MusicVocabulary._get_group2(tok, tpl)
                    return pch-1 + (octave+1)*12  # See `pch2step`, restore the pitch; +1 cos octave starts from -1
            elif typ == VocabType.time_sig:
                return MusicVocabulary._get_group2(tok, tpl)
            elif typ == VocabType.tempo:
                return MusicVocabulary._get_group1(tok, tpl)
            else:
                assert typ == VocabType.key
                return key_str2enum[tpl.match(tok)['key']]

    def uncompact(self, kind: VocabType, compact: Optional[Compact] = None) -> str:
        """
        Reverse operation of `compact`, returns the music "decoded" string
        """
        assert kind != VocabType.special, ValueError(f'Compact representation for special types not supported')
        if kind == VocabType.duration:
            assert isinstance(compact, (int, Tuple[int, int]))
            if isinstance(compact, int):
                return f'{self.cache["pref_dur"]}{compact}'
            else:
                return f'{self.cache["pref_dur"]}{compact[0]}/{compact[1]}'
        elif kind == VocabType.pitch:
            assert isinstance(compact, int)
            if compact == -1:
                return self.cache['rest']
            else:
                pch, octave = compact % 12, compact // 12
                return f'{self.cache["pref_pch"]}{pch}/{octave}'
        elif kind == VocabType.time_sig:
            assert isinstance(compact, tuple)
            return f'{self.cache["pref_time_sig"]}{compact[0]}/{compact[1]}'
        else:  # VocabType.tempo
            assert isinstance(compact, int)
            return f'{self.cache["pref_tempo"]}{compact}'

    @staticmethod
    def pitch_midi2name(midi: int) -> str:
        if midi == -1:
            return 'rest'
        else:
            pch = m21.pitch.Pitch(midi=midi)
            return f'{pch.name}/{pch.octave}'

    def _colorize_spec(self, s: str, color: bool = None) -> str:
        c = self.color if color is None else color
        return log_s(s, c='m') if c else s

    def __getitem__(self, k: str) -> str:
        """
        Index into the special tokens
        """
        return self._colorize_spec(MusicVocabulary.SPEC_TOKS[k])

    def colorize_token(self, tok: str) -> str:
        """
        Colorize token for terminal output
            Color determined by token type
        """
        return log_s(tok, c=MusicVocabulary._token_type2color[self.type(tok)])

    def __call__(
            self, elm: Union[ExtNote, Union[TimeSignature, TsTup], Union[MetronomeMark, int], str],
            color: bool = None,
            return_int: bool = False  # TODO
    ) -> Union[List[str], List[int]]:  # TODO: Support chords?
        """
        Convert music21 element to string or int

        :param elm: A relevant token in melody extraction
        :param color: If given, overrides coloring for current call
        :param return_int: If true, integer ids are returned
        :return List of strings of the converted tokens
        """
        c = self.color if color is None else color

        def colorize(s):
            return self.colorize_token(s) if c else s

        if isinstance(elm, TimeSignature) or (isinstance(elm, tuple) and isinstance(elm[0], int)):  # Time Signature
            if isinstance(elm, TimeSignature):
                top, bot = elm.numerator, elm.denominator
            else:
                top, bot = elm
            return [colorize(self.cache['pref_time_sig']+f'{top}/{bot}')]
        elif isinstance(elm, (int, MetronomeMark)):  # Tempo
            if isinstance(elm, MetronomeMark):
                elm = round(elm.number)  # should be integer
            return [colorize(self.cache['pref_tempo']+str(elm))]
        elif isinstance(elm, Rest):
            r = self.cache['rest']
            return [log_s(r, c='b') if color else r, self._note2dur_str(elm)]
        elif isinstance(elm, Note):
            return [self.note2pitch_str(elm), self._note2dur_str(elm)]
        elif isinstance(elm, tuple):
            # Sum duration for all tuplets
            bot, eot = self.cache['bot'], self.cache['eot']
            return [colorize(bot)] + [
                (self.note2pitch_str(e)) for e in elm
            ] + [self._note2dur_str(elm)] + [colorize(eot)]
        elif isinstance(elm, str):
            assert elm in key_str2enum
            return [colorize(self.cache['pref_key'] + str(elm))]
        else:  # TODO: chords
            ic('other element type', elm)
            exit(1)

    def _note2dur_str(
            self, e: Union[ExtNote, Dur]) -> str:
        """
        :param e: A note, tuplet, or a numeric representing duration
        """
        # If a float, expect multiple of powers of 2
        dur = Fraction(e if isinstance(e, (int, float, Fraction)) else note2dur(e))
        if dur.denominator == 1:
            s = f'{self.cache["pref_dur"]}{dur.numerator}'
        else:
            s = f'{self.cache["pref_dur"]}{dur.numerator}/{dur.denominator}'
        return log_s(s, c='g') if self.color else s

    def note2pitch_str(self, note: Union[Note, Rest, Pitch]) -> str:
        """
        :param note: A note, tuplet, or a music21.pitch.Pitch
        """
        def pch2step(p: Pitch) -> int:
            """
            Naive mapping to the physical, mod-12 pitch frequency, in [1-12]
            """
            return (p.midi % 12) + 1
        if isinstance(note, Rest):
            s = self.cache["rest"]
        else:
            pitch = note.pitch if isinstance(note, Note) else note
            # `pitch.name` follows certain scale by music21 default, may cause confusion
            s = f'{self.cache["pref_pch"]}{pch2step(pitch)}/{pitch.octave}'
        return log_s(s, c='b') if self.color else s

    def _uncommon_tok2uncommon_tok(self, tok: str) -> str:
        typ = self.type(tok)
        assert typ in (VocabType.duration, VocabType.time_sig, VocabType.tempo)  # sanity check
        if typ == VocabType.duration:
            return MusicVocabulary.uncommon_duration
        elif typ == VocabType.time_sig:
            return MusicVocabulary.uncommon_time_sig
        else:  # VocabType.tempo
            tp = self.compact(tok)  # get the actual BPM
            return MusicVocabulary.uncommon_low_tempo if tp < 40 else MusicVocabulary.uncommon_high_tempo

    def clean_uncommon(self, s: str, return_joined: bool = True) -> str:
        """
        Convert uncommon tokens from input score into the special `uncommon` token
        """
        toks = [(tok if tok in self.tok2id else self._uncommon_tok2uncommon_tok(tok)) for tok in s.split()]
        return ' '.join(toks) if return_joined else toks

    def t2i(self, tok):
        if tok in self.tok2id:  # uncommon
            tok = self._uncommon_tok2uncommon_tok(tok)
        return self.tok2id[tok]

    def i2t(self, id_):
        return self.id2tok[id_]

    def encode(self, s: Union[str, List[str], List[List[str]]]) -> Union[int, List[int], List[List[int]]]:
        """
        Convert string token or tokens to integer id
        """
        if isinstance(s, List) and isinstance(s[0], List):
            return list(conc_map(self.encode, s))
        elif isinstance(s, List):
            return [self.tok2id[s_] for s_ in s]
        else:
            return self.tok2id[s]

    def decode(self, id_: Union[int, List[int], List[List[int]]]) -> Union[str, List[str], List[List[str]]]:
        """
        Reverse function of `str2id`
        """
        if isinstance(id_, List) and isinstance(id_[0], List):
            return list(conc_map(self.decode, id_))
        elif isinstance(id_, List):
            return [self.id2tok[i_] for i_ in id_]
        else:
            return self.id2tok[id_]


if __name__ == '__main__':
    from icecream import ic

    ic.lineWrapWidth = 512

    mv = MusicVocabulary()
    # ic(mv.get_durations(exp='dur'))

    # ic(mv.to_dict(save=True))

    def check_vocab_size():
        for k, v in mv.toks.items():
            ic(k, len(v))
        ic(sum(len(v) for v in mv.toks.values()))
    check_vocab_size()

    def check_compact_pitch():
        for i in range(128):
            pch = Pitch(midi=i)
            tok = mv.note2pitch_str(pch)
            ic(i, tok, mv.compact(tok))
            assert i == mv.compact(tok) == pch.midi
    # check_compact_pitch()

    def sanity_check_uncom():
        """
        A small ratio of tokens should be set to uncommon
        """
        from collections import Counter

        import datasets
        from tqdm.auto import tqdm

        from musicnlp.util import sconfig

        np.random.seed(sconfig('random-seed'))

        dnm = 'musicnlp music extraction, dnm=LMD, n=176640, meta={mode=melody, prec=5, th=1}, 2022-05-27_15-23-20'
        path = os_join(music_util.get_processed_path(), 'hf', dnm)
        dsets = datasets.load_from_disk(path)
        # ic(dsets, len(dsets['train']))
        c = Counter()
        n = 4096 * 4
        for split, dset in dsets.items():
            n_dset = len(dset)
            if n_dset > n:
                idxs = np.random.choice(n_dset, size=n, replace=False)
                for i in tqdm(idxs, desc=split):
                    txt = dset[int(i)]['score']
                    c.update(mv.compact(t) for t in txt.split() if mv.has_compact(t))
            else:
                for row in tqdm(dset, desc=split):
                    txt = row['score']
                    c.update(mv.compact(t) for t in txt.split() if mv.has_compact(t))
        ic(c)
    # sanity_check_uncom()
