import re
import json
import math
from os.path import join as os_join
from enum import Enum
from typing import List, Tuple, Set, Dict, Iterator, Optional, Union
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
    'VocabType', 'MusicVocabulary',
    'IsNonRestValidPitch', 'nrp'
]


COMMON_TIME_SIGS: List[TsTup] = sorted(  # Sort first by denominator
    [(4, 4), (2, 4), (2, 2), (3, 4), (6, 8), (5, 4), (12, 8)],
    key=lambda tup_: tuple(reversed(tup_))
)
TEMPO_LOW_EDGE, TEMPO_HIGH_EDGE = 40, 240  # inclusive
COMMON_TEMPOS: List[int] = list(range(TEMPO_LOW_EDGE, TEMPO_HIGH_EDGE+1))  # [40-240]


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
    def with_meta(cls) -> Iterator['VocabType']:
        """
        :return: Iterator of all token types with compact representation
        """
        for i in range(5):  # `special` doesn't have a compact representation
            yield cls(i)


int_types = (int, np.int32, np.int64, torch.Tensor)
Int = Union[int, np.int32, np.int64, torch.Tensor]
TokenMeta = Union[
    TsTup, int, Dur, Key,
    Tuple[None, None], None  # for rare tokens
]


WORDPIECE_CONTINUING_PREFIX = '##'


@dataclass
class PitchNames:
    """
    All possible pitch names for a given pitch ordinal
    """
    normal: List[str] = None
    rare: List[str] = None


@dataclass
class MidiPitchMetaOut:
    """
    Output from midi pitch step expanding
    """
    token: str = None
    local_index: int = None
    octave: int = None


def _get_unique_step_pitch_midis() -> List[int]:
    """
    For the few rare pitches included in the step vocab, with "vanilla" midis outside [0, 127]
        A lazy way to make sure all rare pitches have corresponding tokens in degree vocab

    Basically [-12, *[0, 127], 131]
    """
    vocab = MusicVocabulary(pitch_kind='step')
    toks = vocab.toks['pitch']
    toks = [tok for tok in toks if tok not in (vocab.rest, vocab.rare_pitch)]
    return sorted(set(vocab.pitch_tok2midi_pitch_meta(tok) for tok in toks))


class MusicVocabulary:
    """
    Stores mapping between string tokens and integer ids
    & support the conversion, from relevant `music21` objects to [`str`, `int] conversion
    """
    pad = '[PAD]'  # Needed for type-checking, see `musicnlp.models.metric.get_in_key_ratio`
    # For training sequences subsequences that doesn't start from beginning of song, see `transform::`
    omitted_segment = '[OMIT]'
    start_of_bar = '<bar>'
    start_of_melody = '<melody>'
    start_of_bass = '<bass>'
    end_of_song = '</s>'
    start_of_tuplet = '<tup>'
    end_of_tuplet = '</tup>'

    sep = '_'  # Separation
    time_sig_pref = 'TimeSig'
    tempo_pref = 'Tempo'
    pitch_pref = 'p'
    dur_pref = 'd'
    rare_time_sig = f'{time_sig_pref}{sep}rare'
    rare_low_tempo = f'{tempo_pref}{sep}low'
    rare_high_tempo = f'{tempo_pref}{sep}high'
    rare_pitch = f'{pitch_pref}{sep}rare'  # relevant only for non-midi pitch kind, see `_rarest_index_n_names`
    rare_duration = f'{dur_pref}{sep}rare'  # has to be too long, assuming durations are quantized
    rare_time_sig_meta = (None, None)  # TODO: check, may break things?
    low_tempo_meta = TEMPO_LOW_EDGE - 1  # See `COMMON_TEMPOS`
    high_tempo_meta = TEMPO_HIGH_EDGE + 1
    rare_pitch_meta = None
    rare_duration_meta = None

    special_elm_type2tok = {
        ElmType.seg_omit: omitted_segment,
        ElmType.bar_start: start_of_bar,
        ElmType.melody: start_of_melody,
        ElmType.bass: start_of_bass,
        ElmType.song_end: end_of_song
    }
    midi_rest_pitch_meta = _rest_pitch_meta = -1
    _pitch_kind2rest_pitch_meta = dict(
        midi=_rest_pitch_meta, step=(_rest_pitch_meta, None), degree=(_rest_pitch_meta, None)
    )


    SPEC_TOKS = dict(
        sep=sep,
        rest='r',
        prefix_pitch=pitch_pref,
        prefix_duration=dur_pref,
        omitted_segment=omitted_segment,
        pad=pad,
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
    elm_type2vocab_type = {
        ElmType.time_sig: VocabType.time_sig,
        ElmType.tempo: VocabType.tempo,
        ElmType.key: VocabType.key
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
        11: PitchNames(normal=['A#', 'B-'], rare=[]),  # ['C--', 'G###', 'A--']
        12: PitchNames(normal=['B'], rare=['C-'])  # ['A##']
    }

    # All in `step` pitch:
    # Occurring below 1k times per-token or below 50 per-song across all 3 datasets are considered rarest
    #       & ignored in vocab
    # the per-token and per-song counts are shown
    _rarest_pitch_index_n_names: Set[Tuple[int, str]] = {
        # ======================= Added to rare =======================
        # (6, 'E'),  # 750711/47483
        # (12, 'C'),  # 742230/36617
        # (1, 'B'),  # 542420/29346
        # (5, 'F'),  # 308365/17308
        # (8, 'F'),  # 238836/11837
        # (10, 'B'),  # 106413/6688
        # (3, 'C'),  # 24751/1809
        # (10, 'G'),  # 9324/710

        (11, 'C'),  # 6410/29
        (3, 'E'),  # 736/63
        (5, 'D'),  # 7
    }
    _rarest_pitch_tokens: Set[str] = {  # All found from LMD, pitches at the lowest & highest extremes
        'p_12/10_C',  # 714/1
        'p_8/10_G',  # 156/22
        'p_5/10_E',  # 137/6

        # For the sake of high occurrence of (12, 'C') in other octaves, see `_atonal_pitch_index2name`,
        #   make an exception
        # 'p_12/9_C',  # 111/21

        'p_9/9_A',  # 84/3
        'p_10/9_A',  # 72/4
        'p_6/10_F',  # 64/1
        'p_1/10_C',  # 47/4
        'p_9/9_G',  # 18/2
        'p_7/10_F',  # 13/12
        'p_11/9_A',  # 13/3
        'p_4/10_D',  # 9/2
        'p_1/-2_C',  # 8/2
        'p_1/-3_C',  # 6/1
        'p_11/9_B',  # 6/1
        'p_4/10_E',  # 4/1
        'p_4/-2_E',  # 3/2
        'p_3/-2_D',  # 2/1
        'p_3/10_D',  # 2/2
        'p_10/10_A',  # 2/2
        'p_2/10_C',  # 1/1
        'p_2/-2_D',  # 1/1

        # not sure how this got into extraction output in the first place,
        #   e.g. `C-` with either octave -1 or 0, midi is both 11
        'p_12/-1_C',  # 'p_12/0_C' is in vocab
        'p_1/9_B',  # `p_1/8_B` is in vocab

        # occurrences that justify covering this octave range instead of the other extreme
        # {
        #     'p_12/-1_C': 4/2,  # Not in vocab
        #     'p_12/0_C': 652/4,  # In vocab
        #     'p_12/9_C': 111/7  # In vocab
        # },
        # {
        #     'p_1/-2_B': 0/0,  # In vocab; TODO one token difference...
        #     'p_1/8_B': 338/9,  # In vocab
        #     'p_1/9_B': 1/1,  # Not in vocab
        # }
    }

    # TODO: remove, original training was without key support
    def __init__(
            self, precision: int = 5, color: bool = False, is_wordpiece: bool = False, pitch_kind: str = 'midi',
            with_rare_step: bool = True, tempo_bin: Union[bool, int] = None
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
        :param tempo_bin: If True, tempo is split into groups at regular intervals
        """
        self.precision = precision
        self.color = color
        self.is_wordpiece = is_wordpiece
        ca.check_mismatch('Unique Pitch Kind', pitch_kind, ['midi', 'step', 'degree'])
        self.pitch_kind = pitch_kind
        self.with_rare_step = with_rare_step
        if tempo_bin:
            self.tempo_bin: int = 5 if tempo_bin is True else tempo_bin
        else:
            self.tempo_bin = None
        self.tempo_bin_map, self.tempo_meta2tok, self.tempo_meta_map = None, None, None

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
        self.rest = self.cache['pref_pch'] + specs['rest']

        self._pitch_kind2pattern: Dict[str, re.Pattern] = dict(
            midi=re.compile(rf'^{self.cache["pref_pch"]}{MusicVocabulary.RE2}$'),
            step=re.compile(rf'^{self.cache["pref_pch"]}{MusicVocabulary.RE2}_(?P<step>[A-G])$'),
            degree=re.compile(rf'^{self.cache["pref_pch"]}{MusicVocabulary.RE2}_(?P<step>[1-7])$')
        )
        self._tempo_bin2pattern: Dict[bool, re.Pattern] = {
            True: re.compile(rf'^{self.cache["pref_tempo"]}{MusicVocabulary.RE2}$'),  # See `_get_all_unique_tempos`
            False: re.compile(rf'^{self.cache["pref_tempo"]}{MusicVocabulary.RE1}$')  # No tempo bin, default pattern
        }
        self.tok_type2pattern = {
            VocabType.duration: dict(
                int=re.compile(rf'^{self.cache["pref_dur"]}{MusicVocabulary.RE1}$'),
                frac=re.compile(rf'^{self.cache["pref_dur"]}{MusicVocabulary.RE2}$'),
            ),
            VocabType.pitch: self.pitch_pattern,
            VocabType.time_sig: re.compile(rf'^{self.cache["pref_time_sig"]}{MusicVocabulary.RE2}$'),
            VocabType.tempo: self.tempo_pattern,
            VocabType.key: re.compile(rf'^{self.cache["pref_key"]}(?P<key>.*)$'),
        }

        self.types_with_meta: Set[VocabType] = set(VocabType.with_meta())
        self.rare_tok2meta = {
            MusicVocabulary.rare_time_sig: MusicVocabulary.rare_time_sig_meta,
            MusicVocabulary.rare_low_tempo: MusicVocabulary.low_tempo_meta,
            MusicVocabulary.rare_high_tempo: MusicVocabulary.high_tempo_meta,
            MusicVocabulary.rare_duration: MusicVocabulary.rare_duration_meta
        }
        self.likely_rare_types = (VocabType.pitch, VocabType.duration, VocabType.time_sig, VocabType.tempo)
        # if self.pitch_kind != 'midi':
        # since midi value for some rare pitch falls outside [0-127] range
        self.rare_tok2meta[MusicVocabulary.rare_pitch] = MusicVocabulary.rare_pitch_meta
        self.likely_rare_types = tuple([*self.likely_rare_types, VocabType.pitch])

        def elm2str(elm):
            return self(elm, color=False)

        def rev(time_sig):
            return tuple(reversed(time_sig))  # Syntactic sugar
        tss = [elm2str(rev(ts))[0] for ts in sorted(rev(ts) for ts in COMMON_TIME_SIGS)]
        # See music_visualize.py for distribution
        keys = [elm2str(k)[0] for k in sorted(key_str2enum.keys())]

        special = [specs[k] for k in (
            'omitted_segment',  # TODO: for running earlier models
            'pad', 'start_of_bar', 'end_of_song',
            'start_of_melody', 'start_of_bass', 'start_of_tuplet', 'end_of_tuplet'
        )]
        self.toks: Dict[str, List[str]] = OrderedDict(dict(
            special=special,
            time_sig=[MusicVocabulary.rare_time_sig, *tss],
            tempo=[MusicVocabulary.rare_low_tempo, *self._get_all_unique_tempos(), MusicVocabulary.rare_high_tempo],
            key=keys,
            pitch=self._get_all_unique_pitches(),
            duration=[MusicVocabulary.rare_duration, *self.get_durations(exp='str')]
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
        self.id2meta: Dict[int, TokenMeta] = {
            id_: self.tok2meta(tok) for id_, tok in self.id2tok.items() if self.with_meta(tok)
        }

    def __contains__(self, item: str):
        return item in self.tok2id

    @property
    def tempo_pattern(self) -> re.Pattern:
        return self._tempo_bin2pattern[bool(self.tempo_bin)]

    def _get_all_unique_tempos(self) -> List[str]:
        if self.tempo_bin:
            assert (TEMPO_HIGH_EDGE - TEMPO_LOW_EDGE) % self.tempo_bin == 0
            # in such case, there would be one edge case, last group will have 1 more element
            self.tempo_bin_map: Dict[Tuple[int], Tuple[str, int]] = dict()
            self.tempo_meta_map: Dict[int, int] = dict()  # See `transform.TempoGroup`
            self.tempo_meta2tok: Dict[int, str] = dict()

            bin_strt = TEMPO_LOW_EDGE
            while bin_strt + self.tempo_bin <= TEMPO_HIGH_EDGE:
                bin_end = bin_strt + self.tempo_bin  # exclusive end
                if bin_strt + self.tempo_bin * 2 > TEMPO_HIGH_EDGE:  # last group
                    assert bin_end == TEMPO_HIGH_EDGE  # by construction, for mod is 0
                    bin_end += 1
                key = tuple(range(bin_strt, bin_end))
                # inclusive start & end, separate w/ slash to conform to same format as other tokens
                tok = f'{self.cache["pref_tempo"]}{bin_strt}/{bin_end-1}'
                meta = MusicVocabulary._tempo_bin2meta(start=bin_strt, end=bin_end-1)
                self.tempo_bin_map[key] = (tok, meta)
                self.tempo_meta2tok[meta] = tok

                for tp in key:
                    self.tempo_meta_map[tp] = meta
                bin_strt = bin_end

            # edge meta values stay unchanged
            self.tempo_meta_map[MusicVocabulary.low_tempo_meta] = MusicVocabulary.low_tempo_meta
            self.tempo_meta_map[MusicVocabulary.high_tempo_meta] = MusicVocabulary.high_tempo_meta
            return [tok for tok, _ in self.tempo_bin_map.values()]
        else:
            return [self(tp, color=False)[0] for tp in COMMON_TEMPOS]

    @staticmethod
    def _tempo_bin2meta(start: int = None, end: int = None):
        """
        Inclusive edges
        """
        n = end - start + 1
        return round(sum(range(start, end+1)) / n)  # can't use `tempo_bin` cos last group has 1 more element

    @property
    def rest_pitch_meta(self) -> Union[int, Tuple[int, None]]:
        return MusicVocabulary._pitch_kind2rest_pitch_meta[self.pitch_kind]

    @property
    def pitch_pattern(self) -> re.Pattern:
        return self._pitch_kind2pattern[self.pitch_kind]

    def _get_all_unique_pitches(self) -> List[str]:
        ret = [self.rest, MusicVocabulary.rare_pitch]  # easier code for sanitize rare, TODO
        if self.pitch_kind == 'midi':
            ret += [self.note2pitch_str(Pitch(midi=i)) for i in range(128)]
        else:
            if self.pitch_kind == 'step':
                pchs = []
                for i in range(128):
                    idx = MusicVocabulary.pitch2local_index(i)
                    names = MusicVocabulary._atonal_pitch_index2name[idx]
                    names = names.normal + names.rare if self.with_rare_step else names.normal
                    for name in names:
                        otv = MusicVocabulary.pitch_midi2octave(midi=i)
                        if idx == 1 and name == 'B#':
                            otv -= 1
                        elif idx == 12 and name == 'C-':
                            otv += 1
                        pch = Pitch(name=name, octave=otv)
                        assert pch.midi == i  # sanity check
                        pchs.append(pch)
                ret += [self.note2pitch_str(p) for p in pchs]
            else:  # `degree`
                degs = range(1, 7+1)
                mids = range(128)
                ret += [self.note2pitch_str(Pitch(midi=i), degree=d) for i in mids for d in degs]
        assert len(ret) == len(set(ret))  # sanity check unique
        return ret

    def is_rarest_step_pitch(self, tok: str) -> bool:
        assert self.pitch_kind == 'step'
        mid, step = self.tok2meta(tok, strict=False)
        idx_n_nm = self.pitch2local_index(mid), step
        return idx_n_nm in MusicVocabulary._rarest_pitch_index_n_names or tok in MusicVocabulary._rarest_pitch_tokens

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

    def with_meta(self, tok: Union[str, int]) -> bool:  # See VocabType.with_meta
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

    def tok2meta(self, token: Union[str, Int], strict: bool = True) -> TokenMeta:
        """
        Convert tokens to the numeric format

        More compact, intended for statistics

        Raise error is special tokens passed

        Inverse of `MusicVocabulary.meta2tok`

        :param token: token to compress
        :param strict: If true, enforce that the token must be in the vocabulary
        # :param rare: If true, disable check on vocabulary edge case
        #     Intended for rare pitch sanity check
        :return: If time signature, returns 2-tuple of (int, int),
            If tempo, returns integer of tempo number
            If pitch, returns the pitch MIDI number
            If duration, returns the duration quarterLength

        representation remains faithful to the token for the uncommon ones
        """
        assert self.with_meta(token), ValueError(f'{pl.i(token)} does not have a compact representation')
        if isinstance(token, int_types):
            return self.id2meta[int(token)]
        elif token in self.rare_tok2meta:
            return self.rare_tok2meta[token]
        else:
            typ = self.type(token)
            tpl = self.tok_type2pattern[typ]
            if typ == VocabType.pitch:
                if token == self.rest:
                    return self.rest_pitch_meta
                else:
                    m = self.pitch_pattern.match(token)
                    idx, octave = int(m.group('numer')), int(m.group('denom'))
                    if self.pitch_kind == 'step' and self.with_rare_step:
                        # edge case, see `_get_all_unique_pitches`,
                        # cannot return the midi value in proper range to maintain bijection
                        if octave == -2:
                            assert not strict or token == 'p_1/-2_B'
                            # octave += 1  # this would get the original midi value
                            strict = False
                        elif (idx, octave) == (12, 9):
                            assert not strict or token == 'p_12/9_C'
                            # octave -= 1
                            strict = False
                    mid = idx-1 + (octave+1)*12  # See `pch2step`, restore the pitch; +1 cos octave starts from -1
                    if strict:
                        assert 0 <= mid < 128  # sanity check
                    if self.pitch_kind == 'midi':
                        return mid
                    else:  # `step`, `degree`
                        step = m.group('step')
                        if self.pitch_kind == 'degree':
                            step = int(step)
                        return mid, step
            elif typ == VocabType.duration:
                if '/' in token:
                    numer, denom = MusicVocabulary._get_group2(token, tpl['frac'])
                    assert strict
                    if strict and not math.log2(denom).is_integer():
                        raise ValueError(f'Duration token not quantizable: {pl.i(token)}')
                    # Quantized so definitely an exact float, but keep Fraction for exact additions
                    return Fraction(numer, denom)
                else:
                    return MusicVocabulary._get_group1(token, tpl['int'])
            elif typ == VocabType.time_sig:
                return MusicVocabulary._get_group2(token, tpl)
            elif typ == VocabType.tempo:
                if self.tempo_bin:
                    bin_strt, bin_end = MusicVocabulary._get_group2(token, tpl)  # inclusive
                    return MusicVocabulary._tempo_bin2meta(bin_strt, bin_end)
                else:
                    return MusicVocabulary._get_group1(token, tpl)
            else:
                assert typ == VocabType.key
                return key_str2enum[tpl.match(token)['key']]

    def meta2tok(self, kind: VocabType, meta: Optional[TokenMeta] = None) -> str:
        """
        Reverse operation of `compact`, returns the music "decoded" string
        """
        assert kind != VocabType.special, ValueError(f'Compact representation for special types not supported')
        if kind == VocabType.duration:
            if meta == MusicVocabulary.rare_duration_meta:
                return MusicVocabulary.rare_duration
            else:
                assert isinstance(meta, (int, Fraction))
                if isinstance(meta, int):
                    return f'{self.cache["pref_dur"]}{meta}'
                else:
                    return f'{self.cache["pref_dur"]}{meta.numerator}/{meta.denominator}'
        elif kind == VocabType.pitch:
            if meta == MusicVocabulary.rare_pitch_meta:
                return MusicVocabulary.rare_pitch
            else:
                if self.pitch_kind == 'midi':
                    assert isinstance(meta, int)
                    return self.midi_pitch_meta2tok(meta).token
                else:  # `step`, `degree`
                    assert isinstance(meta, tuple)
                    mid, step = meta
                    assert isinstance(mid, int)  # sanity check

                    out = self.midi_pitch_meta2tok(mid)
                    tok, idx, otv = out.token, out.local_index, out.octave
                    if step is None:
                        assert mid == MusicVocabulary.midi_rest_pitch_meta  # sanity check
                        return tok
                    else:
                        if self.pitch_kind == 'degree':  # sanity check
                            # mic(meta, tok, idx, otv, step)
                            assert isinstance(step, int)
                        else:  # `step`
                            assert isinstance(step, str)
                        return f'{tok}_{step}'
        elif kind == VocabType.time_sig:
            if meta == MusicVocabulary.rare_time_sig_meta:
                return MusicVocabulary.rare_time_sig
            else:
                assert isinstance(meta, tuple)
                return f'{self.cache["pref_time_sig"]}{meta[0]}/{meta[1]}'
        elif kind == VocabType.tempo:
            if meta == MusicVocabulary.low_tempo_meta:
                return MusicVocabulary.rare_low_tempo
            elif meta == MusicVocabulary.high_tempo_meta:
                return MusicVocabulary.rare_high_tempo
            else:
                assert isinstance(meta, int)
                if self.tempo_bin:
                    return self.tempo_meta2tok[meta]
                else:
                    return f'{self.cache["pref_tempo"]}{meta}'
        else:
            assert kind == VocabType.key
            if isinstance(meta, Key):
                meta = enum2key_str[meta]
            return f'{self.cache["pref_key"]}{meta}'

    def midi_pitch_meta2tok(self, meta: int) -> MidiPitchMetaOut:
        if meta == MusicVocabulary.midi_rest_pitch_meta:
            return MidiPitchMetaOut(token=self.rest)
        else:
            pch, octave = meta % 12 + 1, MusicVocabulary.pitch_midi2octave(midi=meta)
            return MidiPitchMetaOut(
                token=f'{self.cache["pref_pch"]}{pch}/{octave}',
                local_index=pch, octave=octave
            )

    def pitch_tok2midi_pitch_tok(self, tok: str, strict: bool = True) -> str:
        assert self.type(tok) == VocabType.pitch
        mid, step = self.tok2meta(tok)
        if strict:  # snap midi in range
            while mid < 0:
                mid += 12
            while mid > 127:
                mid -= 12
        return self.midi_pitch_meta2tok(mid).token

    def pitch_tok2midi_pitch_meta(self, tok: str) -> int:
        """
        Faster implementation to  get midi value from pitch token; TODO: reduce code duplication to `tok2meta`?

        Intended for efficient IKR

        User responsible to make sure a valid pitch is passed in, e.g. not a rest pitch
        """
        m = self.pitch_pattern.match(tok)
        idx, octave = int(m.group('numer')), int(m.group('denom'))
        return idx-1 + (octave+1)*12

    @staticmethod
    def pitch_midi2octave(midi: int) -> int:
        return midi // 12 - 1

    @staticmethod
    def pitch_midi2name(midi: int) -> str:
        if midi == MusicVocabulary.midi_rest_pitch_meta:
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

    def colorize_tokens(self, toks: Union[str, List[str]]) -> str:
        toks = toks if isinstance(toks, list) else toks.split()
        return ' '.join(self.colorize_token(t) for t in toks)

    def __call__(
            self, elm: Union[ExtNote, Union[TimeSignature, TsTup], Union[MetronomeMark, int], Union[str, Key]],
            color: bool = None
    ) -> Union[List[str], List[int]]:  # TODO: Support chords?
        """
        Convert music21 element to string or int

        :param elm: A relevant token in melody extraction
        :param color: If given, overrides coloring for current call
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
            r = self.rest
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
            return [self.meta2tok(MusicVocabulary.elm_type2vocab_type[e.type], e.meta)]
        elif e.type == ElmType.note:
            pch, dur = e.meta
            return [self.meta2tok(VocabType.pitch, pch), self.meta2tok(VocabType.duration, dur)]
        else:
            assert e.type == ElmType.tuplets
            pchs, dur = e.meta
            return [
                self.start_of_tuplet,
                *[self.meta2tok(VocabType.pitch, pch) for pch in pchs],
                self.meta2tok(VocabType.duration, dur),
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
    def pitch2local_index(p: Union[Pitch, int]) -> int:
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
            s = self.rest
        else:
            pitch = note.pitch if isinstance(note, Note) else note
            # `pitch.name` follows certain scale by music21 default, may cause confusion
            s = f'{self.cache["pref_pch"]}{MusicVocabulary.pitch2local_index(pitch)}/{pitch.octave}'
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

    def sanitize_rare_token(self, tok: str, for_midi: bool = False, rare_pitch_only: bool = False) -> str:
        """
        See `musicnlp.preprocess.transform::SanitizeRare`
        """
        if tok in self.tok2id:
            return tok
        else:
            typ = self.type(tok)
            assert typ in self.likely_rare_types  # sanity check
            if typ == VocabType.pitch:
                if for_midi:
                    # to squeeze midi into range [0, 127] that will definitely be part of vocab,
                    # see `transform.ToMidiPitch`
                    mid, step = self.tok2meta(tok, strict=False)
                    while mid < 0:
                        mid += 12
                    while mid > 127:
                        mid -= 12
                    return self.meta2tok(kind=VocabType.pitch, meta=(mid, step))
                else:
                    return MusicVocabulary.rare_pitch
                # return MusicVocabulary.rare_pitch
            elif rare_pitch_only:
                return tok
            else:
                if typ == VocabType.duration:
                    return MusicVocabulary.rare_duration
                elif typ == VocabType.time_sig:
                    return MusicVocabulary.rare_time_sig
                else:
                    assert typ == VocabType.tempo
                    tp = self.tok2meta(tok)  # get the actual BPM
                    return MusicVocabulary.rare_low_tempo if tp < 40 else MusicVocabulary.rare_high_tempo

    def sanitize_rare_tokens(self, s: str, return_as_list: bool = False) -> str:
        """
        Convert uncommon tokens from input score into the special `uncommon` token
        """
        toks = [self.sanitize_rare_token(tok) for tok in s.split()]
        return toks if return_as_list else ' '.join(toks)

    def t2i(self, tok):
        tok = self.sanitize_rare_token(tok)
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


class IsNonRestValidPitch:
    def __init__(self, vocab: MusicVocabulary = None):
        self._vocab = None
        if vocab is not None:
            self._vocab = vocab  # any pitch kind will do

    @property
    def vocab(self) -> MusicVocabulary:
        if self._vocab is None:  # Lazy loading
            self._vocab = MusicVocabulary()
        return self._vocab

    def __call__(self, tok: str) -> bool:
        return self.vocab.type(tok) == VocabType.pitch and tok != self.vocab.rest and tok != self.vocab.rare_pitch


nrp = IsNonRestValidPitch()


if __name__ == '__main__':
    from collections import defaultdict, Counter

    from tqdm.auto import tqdm

    from musicnlp.preprocess import dataset

    mic.output_width = 128

    pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')

    def check_rare_pitch_midi_match(name_rare: str = None, name_norm: str = None, octave: int = 0):
        p_rare = Pitch(name=name_rare, octave=octave)
        p_norm = Pitch(name=name_norm, octave=octave)
        mic(p_rare, p_rare.midi)
        mic(p_norm, p_norm.midi)
        MusicVocabulary(pitch_kind='step')  # make sure pitch creation terminates
    # check_rare_pitch_midi_match(name_rare='B#', name_norm='C', octave=0)  # Raise octave
    # check_rare_pitch_midi_match(name_rare='B--', name_norm='A', octave=2)
    # check_rare_pitch_midi_match(name_rare='A##', name_norm='B', octave=2)
    # check_rare_pitch_midi_match(name_rare='C-', name_norm='B', octave=2)  # Raise octave
    # check_rare_pitch_midi_match(name_rare='A##', name_norm='B', octave=2)
    # check_rare_pitch_midi_match(name_rare='F-', name_norm='E', octave=2)
    # check_rare_pitch_midi_match(name_rare='E#', name_norm='F', octave=2)
    # check_rare_pitch_midi_match(name_rare='F##', name_norm='G', octave=2)
    # check_rare_pitch_midi_match(name_rare='G#', name_norm='A-', octave=2)
    # check_rare_pitch_midi_match(name_rare='D--', name_norm='C', octave=2)
    # check_rare_pitch_midi_match(name_rare='C--', name_norm='B-', octave=2)

    def check_rare_pitch():
        # p = Pitch(name='C-', octave=-1)
        # mic(p, p.midi)
        # mic(MusicVocabulary.pitch2local_index(p.midi))
        # MusicVocabulary(pitch_kind='step')
        # p_op1 = Pitch(name='C-', octave=1)
        # p_o0 = Pitch(name='C-', octave=0)
        # p_o1 = Pitch(name='C-', octave=-1)
        # p_o2 = Pitch(name='C-', octave=-2)
        # p_o3 = Pitch(name='C-', octave=-3)
        # mic(p_op1.octave, p_op1.midi)
        # mic(p_o0.octave, p_o0.midi)
        # mic(p_o1.octave, p_o1.midi)
        # mic(p_o2.octave, p_o2.midi)
        # mic(p_o3.octave, p_o3.midi)

        p_o9 = Pitch(name='B#', octave=9)
        p_o8 = Pitch(name='B#', octave=8)
        p_o7 = Pitch(name='B#', octave=7)
        mic(p_o9.octave, p_o9.midi)
        mic(p_o8.octave, p_o8.midi)
        mic(p_o7.octave, p_o7.midi)
    # check_rare_pitch()
    # exit(1)

    def check_vocab_size():
        mv = MusicVocabulary()
        mic(mv.tok2id)
        for k, v in mv.toks.items():
            mic(k, len(v))
        mic(sum(len(v) for v in mv.toks.values()))
    # check_vocab_size()

    def check_pitch_meta():
        mv = MusicVocabulary()
        for i in range(128):
            pch = Pitch(midi=i)
            tok = mv.note2pitch_str(pch)
            mic(i, tok, mv.tok2meta(tok))
            comp = mv.tok2meta(tok)
            assert i == comp == pch.midi
            tok_ = mv.meta2tok(VocabType.pitch, comp)
            assert tok == tok_
            mic(tok, tok_)
    # check_pitch_meta()

    def check_pitch_set(kind: str = 'step'):
        mv = MusicVocabulary(pitch_kind=kind)
        pchs = mv.toks['pitch']
        mic(pchs, len(pchs))

        tok = 'p_12/9_4'
        mic(tok in mv)
    # check_pitch_set(kind='midi')
    # check_pitch_set(kind='step')
    # check_pitch_set(kind='degree')

    def sanity_check_rare():
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
                    c.update(mv.tok2meta(t) for t in txt.split() if mv.with_meta(t))
            else:
                for row in tqdm(dset, desc=split):
                    txt = row['score']
                    c.update(mv.tok2meta(t) for t in txt.split() if mv.with_meta(t))
        mic(c)
    # sanity_check_rare()

    def check_same_midi_diff_step():

        mv = MusicVocabulary(pitch_kind='step', with_rare_step=False)

        # dnm = '22-10-21_Extracted-POP909_{n=909}_{md=f, prec=5, th=8}'  # Pitch was wrong
        pop = '22-10-22_Extracted-POP909_{n=909}_{md=f, prec=5, th=1}'
        songs = dataset.load_songs(pop)

        counts = defaultdict(Counter)
        for song in tqdm(songs):
            pchs = [tok for tok in song.split() if mv.type(tok) == VocabType.pitch]
            for pch in pchs:
                midi = mv.tok2meta(pch)
                step = pch[-1]
                pch_str = pch[:pch.rfind('_')]
                counts[(pch_str, midi)][step] += 1
        counts = {mid: dict(c) for mid, c in counts.items()}
        mic(counts)
    # check_same_midi_diff_step()

    def get_rare_observed_pitches(per_song: bool = False):
        mic(per_song)

        mv = MusicVocabulary(pitch_kind='step', with_rare_step=False)

        # dnms = [pop]
        # dnms = [pop, mst]
        dnms = [pop, mst, lmd]
        songs = dataset.load_songs(*dnms)

        counts = Counter()
        counts_out_of_range = Counter()
        for song in tqdm(songs, desc='Counting rare pitch'):
            toks = [tok for tok in song.split() if mv.type(tok) == VocabType.pitch and tok not in mv]
            if per_song:
                toks = set(toks)
            for tok in toks:
                mid, step = mv.tok2meta(tok, strict=False)

                if not (0 <= mid < 128):  # No overflow of octave due to the key
                    counts_out_of_range[tok] += 1
                else:
                    idx = MusicVocabulary.pitch2local_index(mid)
                    counts[idx, step] += 1
        mic(counts)

        idx2step = defaultdict(list)
        for (idx, step) in counts.keys():
            idx2step[idx].append(step)
        idx2step = dict(idx2step)
        mic(idx2step)

        mic(counts_out_of_range)
    # get_rare_observed_pitches(per_song=False)
    # get_rare_observed_pitches(per_song=True)

    def check_rare_pitches():
        mv = MusicVocabulary(pitch_kind='step')
        # tok = 'p_3/5_D'
        tok = 'p_5/10_E'
        mic(mv.is_rarest_step_pitch(tok))
        # mic(mv.compact(tok))
    # check_rare_pitches()

    def check_all_step_pitches_covered():
        mv = MusicVocabulary(pitch_kind='step')

        # mic(mv.toks['pitch'])
        # exit(1)

        overlap = set(mv.toks['pitch']) & set(MusicVocabulary._rarest_pitch_tokens)
        assert len(overlap) == 0  # No overlap

        # dnms = [pop]
        # dnms = [pop, mst]
        dnms = [pop, mst, lmd]
        songs = dataset.load_songs(*dnms)

        _count = Counter()  # the 2 included edge case with octave shifting
        _count_per_song = Counter()

        it = tqdm(songs, desc='Checking tokens in vocab')
        for i, song in enumerate(it):
            # if i < 150000:  # Debugging
            #     continue
            toks = (tok for tok in song.split() if mv.type(tok) == VocabType.pitch and tok != mv.rest)

            added = False
            for tok in toks:
                # it.set_postfix(tok=pl.i(tok))  # slows things down
                in_vocab = tok in mv
                rare = mv.is_rarest_step_pitch(tok)
                mut_ex = (in_vocab or rare) and (in_vocab != rare)
                if not mut_ex:
                    mic(tok, in_vocab, rare)
                assert mut_ex  # mutually exclusive
                # able to reconstruct pitch for either case
                comp = mv.tok2meta(token=tok, strict=in_vocab)
                recon = mv.meta2tok(kind=VocabType.pitch, meta=comp)
                if tok != recon:
                    mic(tok, comp, recon)
                    mic(rare)
                assert tok == recon
                if (tok[-1] == 'C' and 'p_12/' in tok) or (tok[-1] == 'B' and 'p_1/' in tok):
                    _count[tok] += 1
                    if not added:
                        _count_per_song[tok] += 1
                    added = True
        mic(_count, _count_per_song)
    # check_all_step_pitches_covered()

    def check_tempo_bin():
        # pch_kd = 'step'
        pch_kd = 'degree'
        mv = MusicVocabulary(pitch_kind=pch_kd, tempo_bin=5)
        mic(len(mv))
        mic(mv.tok2id, mv.id2meta)
        for tok in mv.toks['tempo']:
            b = mv.tok2meta(tok)
            mic(tok, b)
    check_tempo_bin()
