import re
import json
import math
from os.path import join as os_join
from enum import Enum
from typing import List, Tuple, Set, Dict, Iterator, Optional, Union, Literal
from fractions import Fraction
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
import torch
import music21 as m21

from stefutil import *
from musicnlp.util.music_lib import *
import musicnlp.util.music as music_util
from musicnlp.vocab.elm_type import ElmType, MusicElement, Key, key_str2enum, enum2key_str


__all__ = [
    'COMMON_TEMPOS', 'is_common_tempo', 'COMMON_TIME_SIGS', 'is_common_time_sig', 'get_common_time_sig_duration_bound',
    'WORDPIECE_CONTINUING_PREFIX',
    'VocabType', 'MusicVocabulary'
]


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


WORDPIECE_CONTINUING_PREFIX = '##'


@dataclass
class PitchNames:
    """
    All possible pitch names for a given pitch ordinal
    """
    normal: List[str] = None
    rare: List[str] = None


class MusicVocabulary:
    """
    Stores mapping between string tokens and integer ids
    & support the conversion, from relevant `music21` objects to [`str`, `int] conversion
    """
    start_of_bar = '<bar>'
    start_of_melody = '<melody>'
    start_of_bass = '<bass>'
    end_of_song = '</s>'
    start_of_tuplet = '<tup>'
    end_of_tuplet = '</tup>'
    pad = '[PAD]'  # Needed for type-checking, see `musicnlp.models.metric.get_in_key_ratio`

    sep = '_'  # Separation
    time_sig_pref = 'TimeSig'
    tempo_pref = 'Tempo'
    pitch_pref = 'p'
    dur_pref = 'd'
    uncommon_time_sig = f'{time_sig_pref}{sep}uncommon'
    uncommon_low_tempo = f'{tempo_pref}{sep}low'
    uncommon_high_tempo = f'{tempo_pref}{sep}high'
    uncommon_pitch = f'{pitch_pref}{sep}uncommon'  # relevant only for non-midi pitch kind, see `_rarest_index_n_names`
    uncommon_duration = f'{dur_pref}{sep}uncommon'  # has to be too long, assuming durations are quantized
    compact_uncommon_time_sig = (None, None)  # TODO: check, may break things?
    compact_low_tempo = 40-1  # See `COMMON_TEMPOS`
    compact_high_tempo = 240+1
    compact_uncommon_pitch = None
    compact_uncommon_duration = None

    special_elm_type2tok = {
        ElmType.bar_start: start_of_bar,
        ElmType.melody: start_of_melody,
        ElmType.bass: start_of_bass,
        ElmType.song_end: end_of_song
    }
    rest_pitch_code = -1

    SPEC_TOKS = dict(
        sep=sep,
        rest='r',
        prefix_pitch=pitch_pref,
        prefix_duration=dur_pref,
        start_of_tuplet=start_of_tuplet,
        end_of_tuplet=end_of_tuplet,
        start_of_bar=start_of_bar,
        end_of_song=end_of_song,
        prefix_time_sig=time_sig_pref,
        prefix_tempo=tempo_pref,
        prefix_key='Key',
        start_of_melody=start_of_melody,
        start_of_bass=start_of_bass
    )

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

    _atonal_pitch_index2name: Dict[int, PitchNames] = {  # Possible pitch step names given the index
        # for rare ones, keep only those observed in the dataset (and appeared frequently) to save vocab size
        1: PitchNames(normal=['C'], rare=['B#']),
        2: PitchNames(normal=['C#', 'D-'], rare=[]),  # ['B##', 'C--']
        3: PitchNames(normal=['D'], rare=['C##']),  # ['D--']
        4: PitchNames(normal=['D#', 'E-'], rare=[]),  # ['C###', 'D--']
        5: PitchNames(normal=['E'], rare=['F-']),  # ['D##']
        6: PitchNames(normal=['F'], rare=['E#']),
        7: PitchNames(normal=['F#', 'G-'], rare=[]),  # ['E##', 'F--']
        8: PitchNames(normal=['G'], rare=['F##']),
        9: PitchNames(normal=['G#', 'A-'], rare=[]),  # ['F###', 'G--']
        10: PitchNames(normal=['A'], rare=['B--', 'G##']),
        11: PitchNames(normal=['A#', 'B-'], rare=['C--']),  # ['G###', 'A--']
        12: PitchNames(normal=['B'], rare=['C-'])  # ['A##']
    }
    _rarest_index_n_names: Set[Tuple[int, str]] = {  # Occurring below 1k times across all 3 datasets, and the counts
        (3, 'E'),  # 736
        (8, 'G'),  # 156  # TODO: why aren't those in vocab???
        (5, 'E'),  # 137
        (9, 'A'),  # 84
        (10, 'A'),  # 74
        (6, 'F'),  # 64
        (1, 'C'),  # 61
        (9, 'G'),  # 18
        (7, 'F'),  # 13,
        (11, 'A'),  # 13
        (4, 'D'),  # 9
        (5, 'D'),  # 7
        (4, 'E'),  # 7
        (11, 'B'),  # 6
        (3, 'D'),  # 4
        (2, 'C'),  # 1
        (2, 'D'),  # 1
    }

    # TODO: remove, original training was without key support
    def __init__(
            self, precision: int = 5, color: bool = False, is_wordpiece: bool = False, pitch_kind: str = 'midi',
            with_rare_step: bool = True
    ):
        """
        :param precision: See `musicnlp.preprocess.music_extractor`
        :param color: If True, string outputs are colorized
            Update individual coloring of subsequent tokens via `__getitem__`
        :param pitch_kind: Guides the set of uniq pitches considered, one of ['midi`, `step`, `degree`]
            If `midi`, 128 pitches of uniq midi value
            If `step`, 17 uniq pitch names in an octave of 12 midi values,
                intended for music extraction with pitch step name
            If `degree`, 128 pitches and its 7 possible ordinals w.r.t. a key,
                intended for training with scale degree annotation
        :param with_rare_step: If True, rare pitch steps are kept in the vocabulary
            Relevant only when `pitch_kind` is `step`
        """
        self.precision = precision
        self.color = color
        self.is_wordpiece = is_wordpiece
        ca.check_mismatch('Unique Pitch Kind', pitch_kind, ['midi', 'step', 'degree'])
        self.pitch_kind = pitch_kind
        self.with_rare_step = with_rare_step

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

        self._pitch_kind2pattern: Dict[str, re.Pattern] = dict(
            midi=re.compile(rf'^{self.cache["pref_pch"]}{MusicVocabulary.RE2}$'),
            step=re.compile(rf'^{self.cache["pref_pch"]}{MusicVocabulary.RE2}_(?P<step>[A-G])$'),
            degree=re.compile(rf'^{self.cache["pref_pch"]}{MusicVocabulary.RE2}_(?P<step>[1-7])$')
        )
        self.type2compact_re = {
            VocabType.duration: dict(
                int=re.compile(rf'^{self.cache["pref_dur"]}{MusicVocabulary.RE1}$'),
                frac=re.compile(rf'^{self.cache["pref_dur"]}{MusicVocabulary.RE2}$'),
            ),
            VocabType.pitch: self.pitch_pattern,
            VocabType.time_sig: re.compile(rf'^{self.cache["pref_time_sig"]}{MusicVocabulary.RE2}$'),
            VocabType.tempo: re.compile(rf'^{self.cache["pref_tempo"]}{MusicVocabulary.RE1}$'),
            VocabType.key: re.compile(rf'^{self.cache["pref_key"]}(?P<key>.*)$'),
        }

        self.compacts: Set[VocabType] = set(VocabType.compact())
        self.uncom_tok2compact = {
            MusicVocabulary.uncommon_time_sig: MusicVocabulary.compact_uncommon_time_sig,
            MusicVocabulary.uncommon_low_tempo: MusicVocabulary.compact_low_tempo,
            MusicVocabulary.uncommon_high_tempo: MusicVocabulary.compact_high_tempo,
            MusicVocabulary.uncommon_duration: MusicVocabulary.compact_uncommon_duration
        }
        if self.pitch_kind != 'midi':
            self.uncom_tok2compact[MusicVocabulary.uncommon_pitch] = MusicVocabulary.compact_uncommon_pitch

        def elm2str(elm):
            return self(elm, color=False, return_int=False)

        def rev(time_sig):
            return tuple(reversed(time_sig))  # Syntactic sugar
        tss = [elm2str(rev(ts))[0] for ts in sorted(rev(ts) for ts in COMMON_TIME_SIGS)]
        # See music_visualize.py for distribution
        tempos = [elm2str(tp)[0] for tp in COMMON_TEMPOS]
        keys = [elm2str(k)[0] for k in sorted(key_str2enum.keys())]

        # TODO: with music-theory, mod-7 scale degree, vocab size would increase
        # TODO: changed the order of sob & eos and added melody & bass prefix, this will affect prior models trained
        special = [specs[k] for k in (
            'start_of_bar', 'end_of_song', 'start_of_melody', 'start_of_bass', 'start_of_tuplet', 'end_of_tuplet'
        )]
        special.append(MusicVocabulary.pad)
        self.toks: Dict[str, List[str]] = OrderedDict(dict(
            special=special,
            time_sig=[MusicVocabulary.uncommon_time_sig, *tss],
            tempo=[MusicVocabulary.uncommon_low_tempo, *tempos, MusicVocabulary.uncommon_high_tempo],
            key=keys,
            pitch=self._get_all_unique_pitches(),
            duration=[MusicVocabulary.uncommon_duration, *self.get_durations(exp='str')]
        ))
        for toks in self.toks.values():
            assert len(set(toks)) == len(toks)  # sanity check no duplicate tokens
        self.tok2id: Dict[str, int] = {  # back-to0back index as ids
            tok: id_ for id_, tok in enumerate(chain_its(toks for toks in self.toks.values()))
        }
        _tok_added = set()
        _tok_dup = []
        self.id2tok = {v: k for k, v in self.tok2id.items()}
        assert len(self.tok2id) == len(self.id2tok)  # Sanity check: no id collision

        # cache them for efficiency
        self.id2type: Dict[int, VocabType] = {id_: self.type(tok) for id_, tok in self.id2tok.items()}
        self.id2compact: Dict[int, Compact] = {
            id_: self.compact(tok) for id_, tok in self.id2tok.items() if self.has_compact(tok)
        }

        self.logger = get_logger('Music Vocab')

    def __contains__(self, item: str):
        return item in self.tok2id

    @property
    def pitch_pattern(self) -> re.Pattern:
        return self._pitch_kind2pattern[self.pitch_kind]

    def _get_all_unique_pitches(self) -> List[str]:
        ret = [self.cache['rest']]
        if self.pitch_kind == 'midi':
            ret += [self.note2pitch_str(Pitch(midi=i)) for i in range(128)]
        else:
            ret.append(MusicVocabulary.uncommon_pitch)
            if self.pitch_kind == 'step':
                pchs = []
                for i in range(128):
                    idx = MusicVocabulary._pitch2local_index(i)
                    names = MusicVocabulary._atonal_pitch_index2name[idx]
                    names = names.normal + names.rare if self.with_rare_step else names.normal
                    for name in names:
                        otv = MusicVocabulary.pitch_midi2octave(midi=i)
                        if idx == 1 and name == 'B#':
                            otv -= 1
                        elif idx == 12 and name == 'C-':
                            otv += 1
                        elif idx == 11 and name == 'C--':
                            otv += 1
                            # mic(self.note2pitch_str(Pitch(name=name, octave=otv)))
                            # raise NotImplementedError
                        pch = Pitch(name=name, octave=otv)
                        # mic(i, idx, otv, name, pch.midi)
                        assert pch.midi == i  # sanity check
                        pchs.append(pch)
                ret += [self.note2pitch_str(p) for p in pchs]
            else:  # `degree`
                degs = range(1, 7+1)
                ret += [self.note2pitch_str(Pitch(midi=i), degree=d) for i in range(128) for d in degs]
        assert len(ret) == len(set(ret))  # sanity check unique
        return ret

    def is_rarest_step_pitch(self, tok: str) -> bool:
        assert self.pitch_kind == 'step'
        mid, step = self.compact(tok, strict=False)
        idx_n_nm = self._pitch2local_index(mid), step
        mic(idx_n_nm)
        return idx_n_nm in MusicVocabulary._rarest_index_n_names

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
            # `startswith` is slower, see https://stackoverflow.com/q/31917372/10732321
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

    def compact(self, tok: Union[str, Int], strict: bool = True) -> Compact:
        """
        Convert tokens to the numeric format

        More compact, intended for statistics

        Raise error is special tokens passed

        :param tok: token to compress
        :param strict: If true, enforce that the token must be in the vocabulary
        :return: If time signature, returns 2-tuple of (int, int),
            If tempo, returns integer of tempo number
            If pitch, returns the pitch MIDI number
            If duration, returns the duration quarterLength

        representation remains faithful to the token for the uncommon ones
        """
        assert self.has_compact(tok), ValueError(f'{pl.i(tok)} does not have a compact representation')
        if isinstance(tok, int_types):
            return self.id2compact[int(tok)]
        elif tok in self.uncom_tok2compact:
            return self.uncom_tok2compact[tok]
        else:
            typ = self.type(tok)
            tpl = self.type2compact_re[typ]
            if typ == VocabType.pitch:
                if tok == self.cache['rest']:
                    mid = MusicVocabulary.rest_pitch_code
                    return mid if self.pitch_kind == 'midi' else (mid, None)
                else:
                    m = self.pitch_pattern.match(tok)
                    # if not m:
                    #     mic(tok, self.uncom_tok2compact, self.pitch_kind)
                    pch, octave = int(m.group('numer')), int(m.group('denom'))
                    if self.pitch_kind == 'step' and self.with_rare_step:
                        if octave == -2:  # edge case, see `_get_all_unique_pitches`
                            assert tok == 'p_1/-2_B'
                            octave += 1
                        elif (pch, octave) == (12, 9):
                            assert tok == 'p_12/9_C'
                            octave -= 1
                        elif pch == 11 and m.group('step') == 'C':
                            # assert tok == 'p_11/0_C'  # kinda not needed...
                            octave -= 1
                    mid = pch-1 + (octave+1)*12  # See `pch2step`, restore the pitch; +1 cos octave starts from -1
                    if strict:
                        if not (0 <= mid < 128):
                            mic(tok, pch, octave)
                            raise NotImplementedError(f'Pitch {tok} is out of range')
                        assert 0 <= mid < 128  # sanity check
                    if self.pitch_kind == 'midi':
                        return mid
                    else:  # `step`, `degree`
                        return mid, m.group('step')
            elif typ == VocabType.duration:
                if '/' in tok:
                    numer, denom = MusicVocabulary._get_group2(tok, tpl['frac'])
                    assert strict
                    if strict and not math.log2(denom).is_integer():
                        raise ValueError(f'Duration token not quantizable: {pl.i(tok)}')
                    # Quantized so definitely an exact float, but keep Fraction for exact additions
                    return Fraction(numer, denom)
                else:
                    return MusicVocabulary._get_group1(tok, tpl['int'])
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
            assert isinstance(compact, (int, Fraction))
            if isinstance(compact, int):
                return f'{self.cache["pref_dur"]}{compact}'
            else:
                return f'{self.cache["pref_dur"]}{compact.numerator}/{compact.denominator}'
        elif kind == VocabType.pitch:
            if self.pitch_kind == 'midi':
                assert isinstance(compact, int)
                return self._uncompact_midi_pitch(compact)
            else:  # `step`, `degree`
                assert isinstance(compact, tuple)
                mid, step = compact
                assert isinstance(mid, int)  # sanity check

                ret = self._uncompact_midi_pitch(mid)
                if step is None:
                    assert mid == MusicVocabulary.rest_pitch_code  # sanity check
                    return ret
                else:
                    if self.pitch_kind == 'degree':  # sanity check
                        assert isinstance(step, int)
                    else:  # `step`
                        assert isinstance(step, str)
                    return f'{ret}_{step}'
        elif kind == VocabType.time_sig:
            assert isinstance(compact, tuple)
            return f'{self.cache["pref_time_sig"]}{compact[0]}/{compact[1]}'
        elif kind == VocabType.tempo:
            assert isinstance(compact, int)
            return f'{self.cache["pref_tempo"]}{compact}'
        else:
            mic(kind)
            assert kind == VocabType.key
            raise NotImplementedError
            # return f'{self.cache["pref_key"]}{key_enum2str[compact]}'

    def _uncompact_midi_pitch(self, compact: int) -> str:
        if compact == MusicVocabulary.rest_pitch_code:
            return self.cache['rest']
        else:
            pch, octave = compact % 12 + 1, MusicVocabulary.pitch_midi2octave(midi=compact)
            return f'{self.cache["pref_pch"]}{pch}/{octave}'

    def pitch2midi_pitch(self, tok: str) -> str:
        assert self.type(tok) == VocabType.pitch
        mid, step = self.compact(tok)
        return self._uncompact_midi_pitch(mid)

    @staticmethod
    def pitch_midi2octave(midi: int) -> int:
        return midi // 12 - 1

    @staticmethod
    def pitch_midi2name(midi: int) -> str:
        if midi == MusicVocabulary.rest_pitch_code:
            return 'rest'
        else:
            pch = m21.pitch.Pitch(midi=midi)
            return f'{pch.name}/{pch.octave}'

    def _colorize_spec(self, s: str, color: bool = None) -> str:
        c = self.color if color is None else color
        return pl.s(s, c='m') if c else s

    def __getitem__(self, k: str) -> str:
        """
        Index into the special tokens
        """
        return self._colorize_spec(MusicVocabulary.SPEC_TOKS[k])

    def _colorize_token(self, tok: str) -> str:
        return pl.s(tok, c=MusicVocabulary._token_type2color[self.type(tok)])

    def colorize_token(self, tok: str) -> str:
        """
        Colorize token for terminal output
            Color determined by token type
        """
        if self.is_wordpiece:
            toks = tok.replace(WORDPIECE_CONTINUING_PREFIX, '')
            return ' '.join(self._colorize_token(t) for t in toks.split())
        else:
            return self._colorize_token(tok)

    def __call__(
            self, elm: Union[ExtNote, Union[TimeSignature, TsTup], Union[MetronomeMark, int], Union[str, Key]],
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
            return [pl.s(r, c='b') if color else r, self._note2dur_str(elm)]
        elif isinstance(elm, Note):
            return [self.note2pitch_str(elm), self._note2dur_str(elm)]
        elif isinstance(elm, tuple):
            # Sum duration for all tuplets
            bot, eot = self.cache['bot'], self.cache['eot']
            return [colorize(bot)] + [
                (self.note2pitch_str(e)) for e in elm
            ] + [self._note2dur_str(elm)] + [colorize(eot)]
        elif isinstance(elm, (str, Key)):
            if isinstance(elm, str):
                assert elm in key_str2enum
            else:
                elm = enum2key_str[elm]
            return [colorize(self.cache['pref_key'] + str(elm))]
        else:  # TODO: chords
            raise NotImplementedError(f'Unknown element type: {pl.i(elm)} w/ type {pl.i(type(elm))}')

    def music_elm2toks(self, e: MusicElement) -> List[str]:
        if e.type in MusicVocabulary.special_elm_type2tok:
            return [MusicVocabulary.special_elm_type2tok[e.type]]
        elif e.type in (ElmType.time_sig, ElmType.tempo, ElmType.key):
            return self.__call__(e.meta, color=False)
        elif e.type == ElmType.note:
            pch, dur = e.meta
            return [self.uncompact(VocabType.pitch, pch), self.uncompact(VocabType.duration, dur)]
        else:
            assert e.type == ElmType.tuplets
            pchs, dur = e.meta
            return [
                self.start_of_tuplet,
                *[self.uncompact(VocabType.pitch, pch) for pch in pchs],
                self.uncompact(VocabType.duration, dur),
                self.end_of_tuplet
            ]

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
        return pl.s(s, c='g') if self.color else s

    @staticmethod
    def _pitch2local_index(p: Union[Pitch, int]) -> int:
        """
        Naive mapping to the physical, mod-12 pitch frequency in midi, to local atonal pitch index in [1-12]
        """
        midi = p.midi if isinstance(p, Pitch) else p
        return (midi % 12) + 1

    def note2pitch_str(self, note: Union[Note, Rest, Pitch], degree: int = None) -> str:
        """
        :param note: A note, tuplet, or a music21.pitch.Pitch
        :param degree: If given, the scale degree original of the note w.r.t a key

        .. note:: Involves music21 object, may be inefficient, try `uncompact`
        """
        if isinstance(note, Rest):
            s = self.cache["rest"]
        else:
            pitch = note.pitch if isinstance(note, Note) else note
            # `pitch.name` follows certain scale by music21 default, may cause confusion
            s = f'{self.cache["pref_pch"]}{MusicVocabulary._pitch2local_index(pitch)}/{pitch.octave}'
            if self.pitch_kind == 'step':
                s = f'{s}_{pitch.step}'
            elif self.pitch_kind == 'degree':
                if not (isinstance(degree, int) and 1 <= degree <= 7):
                    raise ValueError(f'Invalid degree w/ {pl.i(degree)}, should be in [1, 7]')
                s = f'{s}_{degree}'
        return pl.s(s, c='b') if self.color else s

    def get_pitch_step(self, tok: str) -> str:
        if self.pitch_kind in ['step', 'degree']:
            m = self.pitch_pattern.match(tok)
            step = m.group('step')
            if self.pitch_kind == 'degree':
                step = int(step)
            return step
        else:  # `midi`
            raise ValueError(f'Step is not part of vocabulary for pitch kind {pl.i(self.pitch_kind)}')

    def clean_uncommon_token(self, tok: str) -> str:
        if tok in self.tok2id:
            return tok
        else:
            typ = self.type(tok)
            uncom_types = (VocabType.pitch, VocabType.duration, VocabType.time_sig, VocabType.tempo)  # sanity check
            if self.pitch_kind != 'midi':
                uncom_types = tuple([*uncom_types, VocabType.pitch])

            if typ not in uncom_types:
                mic(tok, typ)
                mic(self.tok2id)
                raise NotImplementedError(f'Token {pl.i(tok)} with type {pl.i(typ)} is not in vocabulary '
                                          f'w/ pitch kind {pl.i(self.pitch_kind)}')
            assert typ in uncom_types
            if typ == VocabType.pitch:
                assert tok == MusicVocabulary.uncommon_pitch
                return MusicVocabulary.uncommon_pitch
            if typ == VocabType.duration:
                assert tok == MusicVocabulary.uncommon_duration
                return MusicVocabulary.uncommon_duration
            elif typ == VocabType.time_sig:
                return MusicVocabulary.uncommon_time_sig
            else:
                assert typ == VocabType.tempo
                tp = self.compact(tok)  # get the actual BPM
                return MusicVocabulary.uncommon_low_tempo if tp < 40 else MusicVocabulary.uncommon_high_tempo

    def clean_uncommon(self, s: str, return_joined: bool = True) -> str:
        """
        Convert uncommon tokens from input score into the special `uncommon` token
        """
        toks = [(tok if tok in self.tok2id else self.clean_uncommon_token(tok)) for tok in s.split()]
        return ' '.join(toks) if return_joined else toks

    def t2i(self, tok):
        if tok not in self.tok2id:  # uncommon
            tok = self.clean_uncommon_token(tok)
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
    from collections import Counter

    from tqdm.auto import tqdm

    from musicnlp.preprocess import dataset

    mic.output_width = 128

    def check_rare_pitch(name_rare: str = None, name_norm: str = None, octave: int = 0):
        from music21 import pitch
        p_rare = pitch.Pitch(name=name_rare, octave=octave)
        p_norm = pitch.Pitch(name=name_norm, octave=octave)
        mic(p_rare, p_rare.midi)
        mic(p_norm, p_norm.midi)
        MusicVocabulary(pitch_kind='step')  # make sure pitch creation terminates
    # check_rare_pitch(name_rare='B#', name_norm='C', octave=0)  # Raise octave
    # check_rare_pitch(name_rare='B--', name_norm='A', octave=2)
    # check_rare_pitch(name_rare='A##', name_norm='B', octave=2)
    # check_rare_pitch(name_rare='C-', name_norm='B', octave=2)  # Raise octave
    # check_rare_pitch(name_rare='A##', name_norm='B', octave=2)
    # check_rare_pitch(name_rare='F-', name_norm='E', octave=2)
    # check_rare_pitch(name_rare='E#', name_norm='F', octave=2)
    # check_rare_pitch(name_rare='F##', name_norm='G', octave=2)
    # check_rare_pitch(name_rare='G#', name_norm='A-', octave=2)
    # check_rare_pitch(name_rare='D--', name_norm='C', octave=2)
    # check_rare_pitch(name_rare='C--', name_norm='B-', octave=2)


    def check_vocab_size():
        mv = MusicVocabulary()
        for k, v in mv.toks.items():
            mic(k, len(v))
        mic(sum(len(v) for v in mv.toks.values()))
    # check_vocab_size()

    def check_compact_pitch():
        mv = MusicVocabulary()
        for i in range(128):
            pch = Pitch(midi=i)
            tok = mv.note2pitch_str(pch)
            mic(i, tok, mv.compact(tok))
            comp = mv.compact(tok)
            assert i == comp == pch.midi
            uncomp = mv.uncompact(VocabType.pitch, comp)
            assert tok == uncomp
            mic(tok, uncomp)
    # check_compact_pitch()

    def check_pitch_set(kind: str = 'step'):
        mv_ = MusicVocabulary(pitch_kind=kind)
        pchs = mv_.toks['pitch']
        mic(pchs, len(pchs))
    # check_pitch_set(kind='step')
    # check_pitch_set(kind='degree')

    def sanity_check_uncom():
        """
        A small ratio of tokens should be set to uncommon
        """
        import datasets

        from musicnlp.util import sconfig

        np.random.seed(sconfig('random-seed'))

        mv = MusicVocabulary()
        dnm = 'musicnlp music extraction, dnm=LMD, n=176640, meta={mode=melody, prec=5, th=1}, 2022-05-27_15-23-20'
        path = os_join(music_util.get_processed_path(), 'hf', dnm)
        dsets = datasets.load_from_disk(path)
        # mic(dsets, len(dsets['train']))
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
        mic(c)
    # sanity_check_uncom()

    def check_same_midi_diff_step():
        from collections import defaultdict

        mv = MusicVocabulary(pitch_kind='step', with_rare_step=False)

        # dnm = '22-10-21_Extracted-POP909_{n=909}_{md=f, prec=5, th=8}'  # Pitch was wrong
        pop = '22-10-22_Extracted-POP909_{n=909}_{md=f, prec=5, th=1}'
        songs = dataset.load_songs(pop)

        counts = defaultdict(Counter)
        for song in tqdm(songs):
            pchs = [tok for tok in song.split() if mv.type(tok) == VocabType.pitch]
            for pch in pchs:
                midi = mv.compact(pch)
                step = pch[-1]
                pch_str = pch[:pch.rfind('_')]
                counts[(pch_str, midi)][step] += 1
        counts = {mid: dict(c) for mid, c in counts.items()}
        mic(counts)
    # check_same_midi_diff_step()

    def get_rare_observed_pitches(per_song: bool = False):
        """
        TODO: ignore the rarest of rare pitches?
        ic| counts: Counter({(6, 'E'): 750711,
                     (12, 'C'): 743055,
                     (1, 'B'): 542420,
                     (5, 'F'): 308365,
                     (8, 'F'): 238836,
                     (10, 'B'): 106413,
                     (3, 'C'): 24751,
                     (10, 'G'): 9324,
                     (11, 'C'): 6410,
                     (3, 'E'): 736,
                     (8, 'G'): 156,
                     (5, 'E'): 137,
                     (9, 'A'): 84,
                     (10, 'A'): 74,
                     (6, 'F'): 64,
                     (1, 'C'): 61,
                     (9, 'G'): 18,
                     (7, 'F'): 13,
                     (11, 'A'): 13,
                     (4, 'D'): 9,
                     (5, 'D'): 7,
                     (4, 'E'): 7,
                     (11, 'B'): 6,
                     (3, 'D'): 4,
                     (2, 'C'): 1,
                     (2, 'D'): 1})
        ic| idx2step: {1: ['B', 'C'], 2: ['C', 'D'], 3: ['C', 'E', 'D'], 4: ['E', 'D'], 5: ['F', 'E', 'D'],
            6: ['E', 'F'], 7: ['F'], 8: ['F', 'G'], 9: ['G', 'A'], 10: ['B', 'G', 'A'], 11: ['C', 'A', 'B'], 12: ['C']}
        """
        from collections import defaultdict

        mv = MusicVocabulary(pitch_kind='step', with_rare_step=False)

        pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')
        # dnms = [pop]
        dnms = [pop, mst]
        # dnms = [pop, mst, lmd]
        songs = dataset.load_songs(*dnms)

        counts = Counter()
        counts_out_of_range = Counter()
        for song in tqdm(songs, desc='Counting rare pitch'):
            toks = [tok for tok in song.split() if mv.type(tok) == VocabType.pitch and tok not in mv]
            if per_song:
                toks = set(toks)
            for tok in toks:
                mid, step = mv.compact(tok, strict=False)

                if not (0 <= mid < 128):  # No overflow of octave due to the key
                    counts_out_of_range[tok] += 1
                else:
                    idx = MusicVocabulary._pitch2local_index(mid)
                    counts[idx, step] += 1
        mic(counts)

        idx2step = defaultdict(list)
        for (idx, step) in counts.keys():
            idx2step[idx].append(step)
        idx2step = dict(idx2step)
        mic(idx2step)

        mic(counts_out_of_range)
    get_rare_observed_pitches(per_song=False)
    # get_rare_observed_pitches(per_song=True)

    def check_rare_pitches():
        mv = MusicVocabulary(pitch_kind='step')
        # tok = 'p_3/5_D'
        tok = 'p_5/10_E'
        mic(mv.is_rarest_step_pitch(tok))
        # mic(mv.compact(tok))
    # check_rare_pitches()
