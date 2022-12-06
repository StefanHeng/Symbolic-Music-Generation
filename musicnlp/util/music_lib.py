"""
Music preprocessing utilities
"""
import math
import re
from copy import deepcopy
from typing import List, Tuple, Dict, Iterator, Iterable, Union
from fractions import Fraction
from collections import namedtuple

import numpy as np
import pandas as pd
import music21 as m21
from music21.meter import TimeSignature
from music21.tempo import MetronomeMark
from music21.note import Note, Rest
from music21.pitch import Pitch
from music21.duration import Duration
from music21.chord import Chord
from music21.stream import Measure, Part, Score
from music21.stream import Voice
import matplotlib.pyplot as plt
import seaborn as sns

from stefutil import *
from musicnlp.util.util import *
from musicnlp.util.project_paths import PKG_NM


KEEP_OBSOLETE = False
if KEEP_OBSOLETE:
    import mido
    from mido import MidiFile
    import pretty_midi
    from pretty_midi import PrettyMIDI
    import librosa
    from librosa import display


__all__ = [
    'TimeSignature', 'MetronomeMark', 'Note', 'Rest', 'Chord', 'Pitch', 'Duration', 'Measure', 'Part', 'Score', 'Voice',
    
    'ExtNote', 'SNote', 'Dur', 'TsTup', 'ordinal2dur_type', 'ScoreExt',
    'time_sig2n_slots',
    'eps', 'is_int', 'is_8th', 'quarter_len2fraction', 'pitch2pitch_cleaned',
    'note2pitch', 'note2dur', 'note2note_cleaned', 'notes2offset_duration',
    'time_sig2bar_dur',
    'TupletNameMeta', 'tuplet_postfix', 'tuplet_prefix2n_note', 'fullname2tuplet_meta',
    'is_drum_track', 'is_empty_bars', 'is_rest',
    'it_m21_elm', 'group_triplets', 'flatten_notes', 'unpack_notes', 'pack_notes', 'unroll_notes', 'fill_with_rest',
    'get_offset', 'get_end_qlen', 'debug_pprint_lst_notes',
    'PrecisionChecker',
    'notes_have_gap', 'notes_overlapping', 'non_tuplet_notes_overlapping',
    'is_notes_pos_duration', 'get_notes_duration', 'is_valid_bar_notes',
    'get_score_skeleton', 'insert_ts_n_tp_to_part', 'make_score'
]

if KEEP_OBSOLETE:
    __all__ += [
        'default_tempo',  'tempo2bpm',
        'MidoUtil', 'PrettyMidiUtil', 'Music21Util'
    ]


logger = get_logger('Music Util')


# Note entity/group as far as music extraction is concerned
ExtNote = Union[Note, Rest, Chord, Tuple[Union[Note, Rest]]]
SNote = Union[Note, Rest, Chord]  # Single note
Dur = Union[float, Fraction]
TsTup = Tuple[int, int]
eps = 1e-8  # for music21 duration equality comparison


TupletNameMeta = namedtuple('TupletNameMeta', field_names=['tuplet_string', 'n_note'])
tuplet_postfix = 'plet'  # Postfix for all tuplets, e.g. `Triplet`, `Quintuplet`
tuplet_prefix2n_note = dict(  # Tuplet prefix => expected number of notes
    Tri=3,
    Quintu=5,
    Nonu=9
)


# support up to certain precision; my power index is off by 2 relative to music21's quarterLength
ordinal2dur_type = ['whole', 'half', 'quarter', 'eighth', '16th', '32nd', '64th', '128th', '256th', '512th', '1024th']

instrs_drum = (
    m21.instrument.BassDrum,
    m21.instrument.BongoDrums,
    m21.instrument.CongaDrum,
    m21.instrument.SnareDrum,
    # m21.instrument.SteelDrum,  # we can work with this as it's `PitchedPercussion`
    m21.instrument.TenorDrum,
)

# Type for extracted music
ScoreExt = Union[Score, List[str], List[int], str]


def time_sig2n_slots(time_sig: m21.meter.TimeSignature, precision: int) -> Tuple[int, int]:
    """
    :return: 2 tuple of (#time slots per beat, #time slots in total)
    """
    denom = time_sig.denominator
    numer = time_sig.numerator
    n_slots_per_beat = (1 / denom / (2 ** -precision))
    assert n_slots_per_beat.is_integer()
    n_slots_per_beat = int(n_slots_per_beat)
    n_slots = int(numer * n_slots_per_beat)
    return n_slots_per_beat, n_slots


def is_int(num: Union[float, Fraction], check_close: Union[bool, float] = True) -> bool:
    if isinstance(num, float):
        if check_close:  # Numeric issue summing Fractions with floats
            eps_ = check_close if isinstance(check_close, float) else eps
            return math.isclose(num, round(num), abs_tol=eps_)
        else:
            return num.is_integer()
    else:
        return num.denominator == 1


def is_8th(d: Dur) -> bool:
    """
    :return If Duration `d` in quarterLength, is multiple of 8th note
    """
    return is_int(d*2)


def quarter_len2fraction(q_len: Dur) -> Fraction:
    """
    :param q_len: A quarterLength value to convert
        Requires one of power of 2
    """
    if isinstance(q_len, float):
        numer, denom = q_len, 1.
        while not (numer.is_integer() and denom.is_integer()):  # Should terminate quick for the expected use case
            numer *= 2
            denom *= 2
        return Fraction(int(numer), int(denom))
    else:
        return q_len


def note2pitch(note: ExtNote):
    if isinstance(note, tuple):  # Triplet, return average pitch
        # Duration for each note not necessarily same duration, for transcription quality
        fs, durs = zip(*[(note2pitch(n__), n__.duration.quarterLength) for n__ in note])
        return np.average(fs, weights=durs)
    elif isinstance(note, Note):
        return note.pitch.frequency
    else:
        assert isinstance(note, Rest)
        return 0  # `Rest` given pitch frequency of 0


def note2dur(note: ExtNote) -> Dur:
    if isinstance(note, tuple):
        return sum(note2dur(nt) for nt in note)
    else:
        return note.duration.quarterLength


def pitch2pitch_cleaned(pitch: Pitch) -> Pitch:
    ret = Pitch(name=pitch.name, octave=pitch.octave)
    assert ret.midi == pitch.midi and ret.step == pitch.step  # the only 2 properties we care about
    return ret


def note2note_cleaned(
        note: ExtNote, q_len=None, offset=None, for_output: bool = False, from_tuplet: bool = False
) -> ExtNote:
    """
    :param note: Note to clean
    :param q_len: Quarter length to set
    :param offset: Offset to set
    :param for_output: If true, set duration for tuplet notes in the proper format,
        so that music21 export sees tuplets as a group
        Expects total duration of tuplet to be quantized
    :param from_tuplet: Flag for note creation for tuplet
        Internally used for duration setting in case `for_output`
    :return: A cleaned version of Note or tuplets with only duration, offset and pitch set
        Notes in tuplets are set with-equal duration given by (`q_len` if `q_len` given, else tuplet total length)
    """
    if q_len is None:
        q_len = note2dur(note)
    if isinstance(note, tuple):
        offset = offset or get_offset(note)
        q_len = quarter_len2fraction(q_len)
        dur_ea = q_len/len(note)
        assert not for_output
        if for_output:  # TODO: seems no longer needed after music21==8.1, and this actually doesn't work
            # 2 to keep on par with quarterLength, 1 more as it seems how music21 works...
            my_ordinal = math.log2(q_len.denominator) + 2 + 2
            assert my_ordinal.is_integer()
            my_ordinal = int(my_ordinal)

            notes: List[SNote] = [note2note_cleaned(n, from_tuplet=True) for n in note]
            # cos directly setting a fraction to Duration would result in error in music21 write, if fraction too small
            dur_ea_tup = m21.duration.Tuplet(
                numberNotesActual=len(notes), numberNotesNormal=q_len.numerator,
                # effectively not using the first 2, but it's fine
                # as multiplying `q_len.numerator` is equivalent as smaller ordinal
                durationNormal=ordinal2dur_type[my_ordinal]
            )
            mic(q_len, dur_ea, dur_ea_tup)
            for n in notes:
                n.duration.appendTuplet(dur_ea_tup)
                mic(n.duration)
        else:
            notes: List[SNote] = [note2note_cleaned(n, q_len=dur_ea) for n in note]
        for i, nt_tup in enumerate(notes):
            notes[i].offset = offset + dur_ea * i
        return tuple(notes)
    dur = Duration(quarterLength=q_len)
    dur_args = dict() if from_tuplet else dict(duration=dur)  # `from_tuplet` only true when `for_output`
    assert isinstance(note, (Note, Rest, Chord))
    if isinstance(note, Note):  # Removes e.g. `tie`s
        nt = Note(pitch=pitch2pitch_cleaned(note.pitch), **dur_args)
    elif isinstance(note, Rest):
        nt = Rest(offset=note.offset, **dur_args)
    else:
        notes = [note2note_cleaned(n) for n in note.notes]
        nt = Chord(notes=notes, offset=note.offset, **dur_args)
    # Setting offset in constructor doesn't seem to work per `music21
    nt.offset = offset if offset is not None else note.offset
    return nt


def notes2offset_duration(notes: Union[List[ExtNote], ExtNote]) -> Tuple[List[float], List[Dur]]:
    if notes:
        if isinstance(notes, list):  # Else, single tuplet notes
            notes = flatten_notes(unroll_notes(notes))
        offsets, durs = zip(*[(n.offset, n.duration.quarterLength) for n in notes])
        return offsets, durs
    else:
        return [], []


_pattern_time_sig_str = re.compile(r'(?P<numer>\d*)/(?P<denom>\d*)')


def time_sig2bar_dur(time_sig: Union[TimeSignature, TsTup, str]) -> float:
    """
    :return: Duration of a bar in given time signature, in quarter length
    """
    if isinstance(time_sig, TimeSignature):
        numer, denom = time_sig.numerator, time_sig.denominator
    elif isinstance(time_sig, str):
        m = _pattern_time_sig_str.match(time_sig)
        numer, denom = int(m.group('numer')), int(m.group('denom'))
    else:
        assert isinstance(time_sig, tuple)
        numer, denom = time_sig
    return numer / denom * 4


def fullname2tuplet_meta(fullname: str) -> TupletNameMeta:
    post, pref2n = tuplet_postfix, tuplet_prefix2n_note
    pref = fullname[:fullname.find(post)].split()[-1]
    tup_str: str = f'{pref}{post}'
    if pref in pref2n:
        n_tup = pref2n[pref]
    else:
        assert pref == 'Tu'  # A generic case, music21 processing, different from that of MuseScore
        # e.g. 'C in octave 1 Dotted 32nd Tuplet of 9/8ths (1/6 QL) Note' makes 9 notes in tuplet
        words = fullname.split()
        word_n_tup = words[words.index(tup_str) + 2]
        n_tup = int(word_n_tup[:word_n_tup.find('/')])
    return TupletNameMeta(tuplet_string=tup_str, n_note=n_tup)


def is_drum_track(part: Part) -> bool:

    """
    :return: True if `part` contains *only* `Unpitched`

    Intended for removing drum tracks in music extraction
    """
    # One pass through `part`, more efficient
    has_unpitched, has_percussion, has_note = False, False, False
    for e in part.recurse():  # Need to look through the entire part to check no Notes
        if isinstance(e, instrs_drum):
            return True  # If part has a drum as instrument, take for granted it's a drum track
        elif isinstance(e, m21.note.Note):
            has_note = True
        elif isinstance(e, m21.percussion.PercussionChord):
            has_percussion = True
        elif isinstance(e, m21.note.Unpitched):
            has_unpitched = True
    return (has_unpitched or has_percussion) and not has_note


def is_empty_bars(bars: Iterable[Measure]):
    def bar2elms(b: Measure):
        def stream2elms(stm: Union[Measure, Voice]):
            return list(chain_its((stm[Note], stm[Rest], stm[Chord])))  # Get all relevant notes
        elms = stream2elms(b)
        if b.hasVoices():
            elms += sum((stream2elms(v) for v in b.voices), start=[])
        return elms
    return all(all(isinstance(e, Rest) for e in bar2elms(b)) for b in bars)


def is_rest(note: ExtNote) -> bool:
    return all(isinstance(n, Rest) for n in note) if isinstance(note, tuple) else isinstance(note, Rest)


def it_m21_elm(stream: Union[Measure, Part, Score, Voice], types=(Note, Rest)):
    """
    Iterates elements in a stream, for those that are instances of that of `type`, in the original order
    """
    if isinstance(stream, (Measure, Voice)):
        return iter(filter(lambda elm: isinstance(elm, types), stream))
    else:
        return iter(filter(lambda elm: isinstance(elm, types), stream.flatten()))


def group_triplets(bar) -> List[ExtNote]:
    """
    Identify triplets from a bar from normal notes & group them

    Expect no `Chord` or `Voice` in bar
    """
    lst = []
    it = it_m21_elm(bar)
    elm = next(it, None)
    while elm:
        if 'Triplet' in elm.fullName:
            elm2, elm3 = next(it, None), next(it, None)
            assert elm2 is not None and elm3 is not None
            assert 'Triplet' in elm2.fullName and 'Triplet' in elm3.fullName
            lst.append((elm, elm2, elm3))
        else:
            lst.append(elm)
        elm = next(it, None)
    return lst


def flatten_notes(notes: Iterable[ExtNote]) -> Iterator[SNote]:
    """
    Expand the intermediate grouping of tuplets
    """
    for n in notes:
        if isinstance(n, tuple):
            for n_ in n:
                yield n_
        else:
            yield n


def unpack_notes(
        notes: List[ExtNote]
) -> Tuple[List[SNote], object]:
    """
    :param notes: Notes representation with tuplets in tuples
    :return: 2-tuple of flattened notes, and a cache for reconstructing the packed tuple representation

    .. note:: See `pack_notes`
    """
    return (
        list(flatten_notes(notes)),
        # Stores the starting index & length of each tuple
        {idx: len(notes[idx]) for idx in range(len(notes)) if isinstance(notes[idx], tuple)}
    )


def pack_notes(notes: List[SNote], cache: object) -> List[ExtNote]:
    """
    Reconstructs original note with tuplets in tuples from cache
    """
    notes_out = []
    cache: Dict[int, int]
    if cache:
        it = iter(notes)
        idx = 0
        n_notes = len(notes)
        while idx < n_notes:
            if idx in cache:
                tups = []
                for _ in range(cache[idx]):
                    tups.append(next(it))
                    idx += 1
                notes_out.append(tuple(tups))
            else:
                notes_out.append(next(it))
                idx += 1
        return notes_out
    else:  # Cache empty, i.e. no tuplets
        return notes


def unroll_notes(notes: List[ExtNote]) -> List[ExtNote]:
    """
    :param notes: individual notes with offsets not back-to-back
    :return: Notes as if jointed in time together

    .. note:: Original notes unmodified
    """
    if not notes_overlapping(notes):
        return notes
    else:
        notes_ = list(flatten_notes(notes))
        notes_ = [deepcopy(n) for n in notes_]
        offsets = [0]
        strt = notes_[0].duration.quarterLength
        for note in notes_[1:]:
            offsets.append(strt)
            strt += note.duration.quarterLength
        offsets = iter(offsets)
        for i, note in enumerate(notes):
            if isinstance(note, tuple):
                notes_tup: List[SNote] = list(note)
                for idx, n in enumerate(notes_tup):
                    # If `Chord`, notes inside with offset of 0 always, leave unchanged
                    notes_tup[idx].offset = next(offsets)
                notes[i] = tuple(notes_tup)
            else:
                notes[i].offset = next(offsets)
        return notes


def get_end_qlen(note: ExtNote):
    """
    :return: Ending time in quarter length
    """
    if isinstance(note, tuple):
        return get_end_qlen(note[-1])
    else:
        return note.offset + note.duration.quarterLength


def debug_pprint_lst_notes(notes: List[ExtNote], return_meta=False):
    ret = []
    for n in notes:
        strt, end = get_offset(n), get_end_qlen(n)
        p = n.pitch.nameWithOctave if isinstance(n, Note) else None
        if return_meta:
            ret.append(dict(note=n, start=strt, end=end, pitch=p))
        else:
            mic(n, strt, end, p)
    return ret


class PrecisionChecker:
    def __init__(self, precision: int = 5):
        self.prec = precision
        self.dur_slot = 4 / 2 ** precision  # quarterLength by quantization precision

    def _val_within_prec(self, val: float) -> bool:
        return (val / self.dur_slot).is_integer()

    def note_within_prec(self, note: ExtNote):
        return self._val_within_prec(note2dur(note)) and self._val_within_prec(get_offset(note))

    def notes_within_prec(self, notes: Iterable[ExtNote]):
        return all(self.note_within_prec(n) for n in notes)


def get_offset(note: ExtNote) -> float:
    """
    :return: Starting time in quarter length
    """
    if isinstance(note, tuple):
        return get_offset(note[0])
    else:
        return note.offset


def fill_with_rest(
        notes: Iterable[ExtNote], serializable: bool = True, duration: Dur = None, fill_start: bool = False
) -> Tuple[List[ExtNote], List[Tuple[float, float]]]:
    """
    Fill the missing time with rests

    :param notes: List of notes in non-overlapping sequential order
        Intended for filling quantized notes during music extraction
            This process is already part of `notes2quantized_notes`
    :param serializable: If True, `Fraction`s in unfilled ranges will be converted to string
        Intended for json output
    :param duration: If given, notes are filled up until this ending time
    :param fill_start: If true, fill potentially-mussing starting time with rest
    :return: 2 tuple of (notes with rests added, list of 2-tuple of (start, end) of missing time)

    .. note:: Tuplet group is treated as a whole
    """
    def get_rest_note(strt, end):
        r = Rest(duration=Duration(quarterLength=end-strt))
        r.offset = strt
        return r
    it = iter(notes)
    note = next(it, None)
    if note is None:
        return [get_rest_note(0, duration)], [(0, duration)]
    else:
        lst, meta = [note] if note else [], []
        last_end = get_end_qlen(note)

        def fill(strt, end):
            lst.append(get_rest_note(strt, end))
            if serializable:
                strt, end = serialize_frac(strt), serialize_frac(end)
            meta.append((strt, end))
        note = next(it, None)
        while note is not None:
            new_begin = get_offset(note)
            assert new_begin-last_end >= -eps  # verify input, allow a small eps
            if new_begin-last_end > eps:  # Found gap
                fill(last_end, new_begin)
            lst.append(note)

            last_end = get_end_qlen(note)  # prep for next iter
            note = next(it, None)
        if duration:
            diff = duration - last_end
            if diff - eps > 0:
                fill(last_end, duration)
        if fill_start:
            n0 = lst[0]
            if isinstance(n0, tuple):
                n0 = n0[0]
            end_ = n0.offset
            if end_ != 0:
                lst.insert(0, get_rest_note(0, end_))
                if serializable:
                    end_ = serialize_frac(end_)
                meta.insert(0, (0, end_))
        return lst, meta


def notes_have_gap(notes: Iterable[ExtNote], enforce_no_overlap: bool = True, duration: Dur = None) -> bool:
    it = flatten_notes(notes)
    note = next(it, None)
    if note is None:  # no note at all
        return duration > 0
    else:
        last_end = get_end_qlen(note)
        note = next(it, None)
        while note is not None:
            new_begin = get_offset(note)
            diff = new_begin - last_end
            if enforce_no_overlap and diff+eps < 0:
                raise ValueError(f'Notes overlap: Last note ends at {pl.i(last_end)}, '
                                 f'current note {pl.i(note)} starts at {pl.i(new_begin)}')
            if diff - eps > 0:
                return True
            last_end = get_end_qlen(note)
            note = next(it, None)
        if duration and (duration - last_end - eps) > 0:
            return True
        return False


def notes_overlapping(notes: Iterable[ExtNote], flatten: bool = True) -> bool:
    """
    :return True if notes overlap, given the start time and duration
    """
    notes = flatten_notes(notes) if flatten else iter(notes)
    note = next(notes, None)
    if note is None:
        return False
    else:
        end = get_end_qlen(note)
        note = next(notes, None)
        while note is not None:
            # Current note should begin, after the previous one ends
            # Since numeric representation of one-third durations, aka tuplets
            if (end-eps) <= get_offset(note):
                end = get_end_qlen(note)
                note = next(notes, None)
            else:
                return True
        return False


def _tup2note(t: Tuple[Note]):
    note = Note()
    note.offset = min(note_.offset for note_ in t)
    q_len_max = max(note_.offset + note_.duration.quarterLength for note_ in t) - note.offset
    note.duration = Duration(quarterLength=q_len_max)
    return note


def non_tuplet_notes_overlapping(notes: Iterable[ExtNote]) -> bool:
    # Convert tuplet to single note by duration, pitch doesn't matter, prep for overlap check, see `_tup2note`
    notes_cleaned = [_tup2note(n) if isinstance(n, tuple) else n for n in notes]
    return notes_overlapping(notes_cleaned)


def is_notes_pos_duration(notes: Iterable[ExtNote]) -> bool:
    return all(note.duration.quarterLength > 0 for note in flatten_notes(notes))


def get_notes_duration(notes: Iterable[ExtNote]) -> Dur:
    """
    Assumes notes are in sequential order and no overlap
    """
    ret = sum(n.duration.quarterLength for n in flatten_notes(notes))
    if is_int(ret):
        ret = round(ret)
    return ret


def is_valid_bar_notes(
        notes: Iterable[ExtNote], time_sig: Union[TimeSignature, TsTup, str], check_match_time_sig: bool = True
) -> bool:
    # Ensure notes cover the entire bar; For addition between `float`s and `Fraction`s
    pos_dur = is_notes_pos_duration(notes)
    no_ovl = not notes_overlapping(notes)
    have_gap = notes_have_gap(notes)
    valid = pos_dur and no_ovl and (not have_gap)
    if check_match_time_sig:
        dur_bar = time_sig2bar_dur(time_sig)
        match_bar_dur = math.isclose(get_notes_duration(notes), dur_bar, abs_tol=eps)
        valid = valid and match_bar_dur
    return valid


def get_score_skeleton(title: str = None, composer: str = PKG_NM, mode: str = 'melody') -> Score:
    """
    :return: A `Score` skeleton with title, composer as metadata and a single piano `Part`
    """
    assert mode in ['melody', 'full']
    score = Score()
    score.insert(m21.metadata.Metadata())
    post = 'Melody only' if mode == 'melody' else 'Melody & Bass'
    title = f'{title}, {post}'
    score.metadata.title = title
    score.metadata.composer = composer

    def get_part(pnm: str):
        part = m21.stream.Part(partName=pnm)
        part.partName = pnm
        instr = m21.instrument.Piano()
        part.append(instr)
        return part

    part_melody = get_part('Melody, Ch#1')
    score.append(part_melody)
    if mode == 'full':
        part_bass = get_part('Bass, Ch#2')
        score.append(part_bass)
    return score


def insert_ts_n_tp_to_part(part: Part, time_sig: str, tempo: int) -> Part:
    bar0 = part.measure(0)  # Insert metadata into 1st bar
    bar0.insert(m21.tempo.MetronomeMark(number=tempo))
    bar0.insert(TimeSignature(time_sig))
    return part


def make_score(
        title: str = f'{PKG_NM} Song', composer: str = PKG_NM, mode: str = 'melody',
        time_sig: str = '4/4', tempo: int = 120, d_notes: Dict[str, List[List[SNote]]] = None,
        check_duration_match: str = None
) -> Score:
    """
    :param title: Title of the score
    :param composer: Composer of the score
    :param mode: 'melody' or 'full'
    :param time_sig: Time signature of the score
    :param tempo: Tempo of the score
    :param d_notes: Dict of notes to insert into the score
        Each note would have offset 0
    :param check_duration_match: Check if notes cover the entire bar
        If don't check, and not a duration match, the starting offset for each bar can be wrong
    """
    score = get_score_skeleton(title=title, composer=composer, mode=mode)
    parts = list(score.parts)
    assert len(parts) <= 2  # sanity check, see `get_score_skeleton`
    part_melody = parts[0]
    assert 'Melody' in part_melody.partName
    check_dur = check_duration_match is not None and check_duration_match is not False
    if check_dur:
        ca.check_mismatch('Check Bar Duration Scheme', check_duration_match, ['time-sig', 'each-other'])

    def get_bars(lst_notes: List[List[SNote]], is_bass: bool = False) -> List[Measure]:
        lst_bars = []
        for i, notes in enumerate(lst_notes):
            _notes = []
            for n in notes:
                q_len = n.duration.quarterLength
                if q_len is not None and q_len > 0:
                    _notes.append(n)
                else:  # TODO: insert as rest somewhere in between?
                    logger.warning(f'Invalid duration {pl.i(q_len)} found from note {pl.i(n)} at bar#{pl.i(i)}, '
                                   f'skipping... ')
            notes = _notes

            if check_dur:
                assert all(n.offset == 0 for n in notes)  # sanity check

            if check_duration_match == 'time-sig':
                assert time_sig is not None and time_sig != 'TimeSig_rare'  # sanity check
                dur_notes, dur_bar = get_notes_duration(notes), time_sig2bar_dur(time_sig)
                diff = dur_notes-dur_bar
                if abs(diff) > eps:
                    if len(notes) == 0:
                        logger.warning(f'No notes found at bar {pl.i(i+1)}')

                    d_log = dict(notes_total_duration=dur_notes, bar_duration=dur_bar)
                    typ = 'Bass' if is_bass else 'Melody'
                    msg = f'{pl.i(typ)} notes duration don\'t match time signature duration at bar {pl.i(i + 1)} w/ {pl.i(d_log)}'

                    if dur_notes < dur_bar:
                        dur = dur_bar - dur_notes
                        notes = notes + [Rest(quarterLength=dur)]
                        msg = f'{msg}, rest added of duration {pl.i(dur)}'
                    else:
                        idx_last = None  # Find the first note that's beyond the edge of time sig
                        dur = 0
                        for i_, n in enumerate(notes):
                            dur += n.duration.quarterLength
                            if (dur-dur_bar)-eps > 0:
                                idx_last = i_
                                break
                        assert idx_last is not None

                        dur_prior = get_notes_duration(notes[:idx_last])

                        n_ = len(notes)
                        if abs(dur_prior - dur_bar) < eps:
                            # just drop the notes and duration will be roughly the same
                            notes = notes[:idx_last]
                            msg = f'{msg}, {pl.i(n_ - idx_last)} notes dropped '
                        else:  # TODO: verify
                            ori_qlen = get_end_qlen(notes[idx_last])

                            # Crop duration of the new last note
                            qlen = dur_bar - dur_prior
                            assert qlen > 0  # sanity check
                            notes[idx_last] = note2note_cleaned(notes[idx_last], q_len=qlen)
                            notes = notes[:idx_last+1]

                            n_drop = n_ - idx_last - 1
                            nt_str = 'notes' if n_drop > 1 else 'note'
                            msg = f'{msg}, {pl.i(n_drop)} {nt_str} dropped, last note duration cropped: ' \
                                  f'{pl.i(ori_qlen)} => {pl.i(qlen)}'
                        assert abs(get_notes_duration(notes)-time_sig2bar_dur(time_sig)) < eps  # sanity check
                    logger.warning(msg)
            bar = Measure(number=i)  # Original bar number may not start from 0
            bar.append(notes)
            if is_bass and i == 0:
                bar.insert(m21.clef.BassClef())
            lst_bars.append(bar)
        return lst_bars
    lst_notes_melody = get_bars(d_notes['melody'])
    lst_notes_bass = None
    if mode == 'full':
        lst_notes_bass = get_bars(d_notes['bass'], is_bass=True)
        if check_duration_match == 'each-other':
            for idx, (notes_m, notes_b) in enumerate(zip(lst_notes_melody, lst_notes_bass)):
                dur_m, dur_b = get_notes_duration(notes_m), get_notes_duration(notes_b)
                d = dict(melody=dur_m, bass=dur_b)

                if abs(dur_m-dur_b) > eps:
                    if dur_m > dur_b:
                        gap = dur_m - dur_b
                        notes_b.append(Rest(quarterLength=gap))
                    else:
                        gap = dur_b - dur_m
                        notes_m.append(Rest(quarterLength=gap))
                    logger.warning(f'Melody and bass notes duration don\'t match at bar {pl.i(idx+1)} '
                                   f'w/ durations {pl.i(d)}')

    part_melody.append(lst_notes_melody)
    bar0 = part_melody.measure(0)  # Insert metadata into 1st bar
    bar0.insert(MetronomeMark(number=tempo))
    if time_sig is not None and time_sig != 'TimeSig_rare':  # so that edge case runs...
        bar0.insert(TimeSignature(time_sig))

    if mode == 'full':
        part_bass = parts[1]  # see `get_score_skeleton`
        assert 'Bass' in part_bass.partName  # sanity check
        part_bass.append(lst_notes_bass)

        # sanity check
        offsets_m = [bar.offset for bar in part_melody[Measure]]
        offsets_b = [bar.offset for bar in part_bass[Measure]]
        for _i, (o_m, o_b) in enumerate(zip(offsets_m, offsets_b)):
            if o_m != o_b:
                raise ValueError(f'Offset mismatch: {pl.i(_i)}, {pl.i(o_m)}, {pl.i(o_b)}')
        assert offsets_m == offsets_b  # sanity check
    return score


if KEEP_OBSOLETE:
    default_tempo = int(5e5)  # Midi default tempo (ms per beat, i.e. 120 BPM)


    def tempo2bpm(tempo):
        """
        :param tempo: ms per beat
        :return: Beat per minute
        """
        return 60 * 1e6 / tempo


    class MidoUtil:
        def __init__(self):
            pass

        @staticmethod
        def get_msgs_by_type(fl, t_, as_dict=False):
            def _get(track):
                return list(filter(lambda tr: tr.type == t_, track))
            if type(fl) is MidiFile or type(fl) is str:
                if type(fl) is str:
                    fl = MidiFile(fl)
                d_msgs = {i: _get(trk) for i, trk in enumerate(fl.tracks)}
                return d_msgs if as_dict else flatten(d_msgs.values())
            elif type(fl) is mido.MidiTrack:
                return _get(fl)

        @staticmethod
        def get_tempo_changes(mf, dedupe=False):
            lst_msgs = MidoUtil.get_msgs_by_type(mf, 'set_tempo')
            if lst_msgs:
                tempos = [msg.tempo for msg in lst_msgs]
                return list(set(tempos)) if dedupe else tempos
            else:
                return [default_tempo]


    class PrettyMidiUtil:
        def __init__(self):
            pass

        @staticmethod
        def plot_single_instrument(instr_, cols=('start', 'end', 'pitch', 'velocity'), n=None):
            """
            Plots a single `pretty_midi.Instrument` channel

            :return: pd.DataFrame of respective attributes in the note, up until `n`
            """
            def _get_get(note):
                return [getattr(note, attr) for attr in cols]
            df = pd.DataFrame([_get_get(n) for n in instr_.notes], columns=cols)[:n]
            df.plot(
                figsize=(16, 9),
                lw=0.25, ms=0.3,
                title=f'Instrument notes plot - [{",".join(cols)}]'
            )
            return df

        @staticmethod
        def get_pitch_range(pm_, clip_=False):
            """
            :return: Inclusive lower and upper bound of pitch in PrettyMIDI
            """
            def _get_pitch_range():
                def _get(instr_):
                    arr = np.array([n.pitch for n in instr_.notes])
                    return np.array([arr.min(), arr.max()])

                if type(pm_) is PrettyMIDI:
                    ranges = np.vstack([_get(i) for i in pm_.instruments])
                    return ranges[:, 0].min(), ranges[:, 1].max()
                else:
                    return list(_get(pm_))
            strt, end = _get_pitch_range()
            if clip_:  # clip by octave
                strt = math.floor(strt / 12) * 12
                end = math.ceil(end / 12) * 12
            return strt, end

        @staticmethod
        def plot_piano_roll(fl, strt=None, end=None, fqs=100, with_beats=True):
            """
            :param fl: A PrettyMIDI, Instrument, or file path string
            :param strt: Lowest pitch to plot inclusive, inferred if not given
            :param end: Highest pitch to plot inclusive, inferred if not given
            :param fqs: Sample frequency
            :param with_beats: If true, beat times are shown as vertical lines
                If fl is Instrument, expect the times needs to be provided
            """
            if type(fl) is str:
                fl = PrettyMIDI(fl)
            if strt is None and end is None:
                strt, end = PrettyMidiUtil.get_pitch_range(fl, clip_=True)
                end += 1  # inclusive np array slicing
            pr_ = fl.get_piano_roll(fqs)[strt:end]
            strt_ = pretty_midi.note_number_to_name(strt)
            end_ = pretty_midi.note_number_to_name(end)

            plt.figure(figsize=(16, (9 * (end-strt+1) / 128) + 1), constrained_layout=True)
            kwargs = dict(
                fmin=pretty_midi.note_number_to_hz(strt) if strt else None
            )
            with sns.axes_style('ticks'):
                librosa.display.specshow(
                    pr_,
                    hop_length=1, sr=fqs, x_axis='time', y_axis='cqt_note',
                    cmap='mako',
                    **kwargs
                )
            nms = np.arange(strt, end, 6)
            plt.colorbar(pad=2**(-5))
            plt.yticks(
                list(map(pretty_midi.note_number_to_hz, nms)),
                list(map(pretty_midi.note_number_to_name, nms))
            )
            if (type(with_beats) is bool and with_beats) or with_beats is not None:
                ticks = plt.yticks()[0]
                x = fl.get_beats() if type(fl) is PrettyMIDI else with_beats
                ax = plt.gca()
                ax.vlines(x=x, ymin=ticks.min(), ymax=ticks.max(), lw=0.25, alpha=0.5, label='Beats')

            plt.title(f'Piano roll plot - [{strt_}, {end_}]')
            plt.show()


    class Music21Util:
        def __init__(self):
            pass

        @staticmethod
        def plot_piano_roll(stream, title=None, s=0, e=None):
            """
            :return: music21 graph object for plotting
            """
            mess = stream.measures(s, e)
            plt_ = m21.graph.plot.HorizontalBarPitchSpaceOffset(
                mess, figsize=(16, 9), constrained_layout=False, doneAction=None
            )
            plt_.colors = sns.color_palette(palette='husl', n_colors=2**4)
            plt_.fontFamily = 'sans-serif'
            plt_.run()

            plt.tight_layout()
            x_ticks = plt.xticks()[0]
            y_ticks = plt.yticks()[0]
            x_d, y_d = np.diff(x_ticks).max(), np.diff(y_ticks).max()
            offset_x, offset_y = x_d / 2**3, y_d / 2**1
            x_s, x_e = x_ticks[0], x_ticks[-1] + x_d
            y_s, y_e = y_ticks[0], y_ticks[-1]
            plt.xlim([x_s - offset_x, x_e + offset_x])
            plt.ylim([y_s - offset_y, y_e + offset_y])

            fig = plt.gcf()
            fig.set_size_inches(16, clip(9 * (y_e - y_s) / (x_e - x_s), 16 / 2 ** 3, 16 / 2))
            strt = s
            end = len(stream.measures(0, e)) if e is None else e
            title = (
                    title or
                    (stream.metadata and stream.metadata.title) or
                    (stream.activeSite and stream.activeSite.metadata and stream.activeSite.metadata.title) or
                    stream.id
            )
            if isinstance(stream, m21.stream.Part) and stream.partName:
                title = f'{title}, {stream.partName}'
            plt.title(f'Piano roll, {title}, bars {strt}-{end}')
            txt_prop = dict(fontname='DejaVu Sans')
            plt.yticks(*plt.yticks(), **txt_prop)
            plt.show()
            return plt_


if __name__ == '__main__':
    from music21 import graph

    mic.output_width = 256

    import musicnlp.util.music as music_util

    # mic(tempo2bpm(DEF_TPO))

    def check_note2hz():
        for n in np.arange(0, 12*10, 12):
            mic(n, pretty_midi.note_number_to_hz(n))
    # check_note2hz()

    def check_piano_roll():
        # pm = pretty_midi.PrettyMIDI(eg_midis('Shape of You'))
        pm = pretty_midi.PrettyMIDI(music_util.get_my_example_songs('Merry Go Round of Life'))

        # pr = pm.get_piano_roll(100)
        # mic(pr.shape, pr.dtype, pr[75:80, 920:960])
        # # mic(np.where(pr > 100))
        #
        # instr0 = pm.instruments[0]
        # instr1 = pm.instruments[1]
        # mic(instr0.get_piano_roll()[76, 920:960])
        # mic(instr1.get_piano_roll()[76, 920:960])

        pmu = PrettyMidiUtil()
        pmu.plot_piano_roll(pm, fqs=100)
        # pmu.plot_piano_roll(instr0)
    # check_piano_roll()

    def test_show_in_plot():
        data = [('Chopin', [(1810, 1849 - 1810)]),
                ('Schumanns', [(1810, 1856 - 1810), (1819, 1896 - 1819)]),
                ('Brahms', [(1833, 1897 - 1833)])]
        xTicks = [(1810, '1810'),
                  (1848, '1848'),
                  (1897, '1897')]
        ghb = graph.primitives.GraphHorizontalBar()
        ghb.title = 'Romantics live long and not so long'
        ghb.data = data
        ghb.setTicks('x', xTicks)

        ghb.doneAction = None  # The `show` action calls figure.show() which doesn't seem to work in Pycharm
        ghb.process()
        # ghb.callDoneAction()
        plt.show()
    # test_show_in_plot()

    def test_piano_roll():
        fnm = music_util.get_my_example_songs('Merry Go Round of Life', fmt='MXL')
        mic(fnm)

        scr = m21.converter.parse(fnm)
        mic(scr.id)
        # Looks like this doesn't work **sometimes**?
        # r = scr.plot('pianoroll', figureSize=(16, 9), doneAction=None)
        # mic(r)

        part = scr.parts[0]
        # plt_ = graph.plot.HorizontalBarPitchSpaceOffset(part.measures(25, 90), doneAction=None, figsize=(16, 9))
        # plt_.run()
        # mic(len(plt_.data), plt_.data)
        # plt.tight_layout()
        # mic(plt.xlim(), plt.ylim())
        # mic(plt.xticks(), plt.yticks())
        # # plt.xlim([0, 501])
        # plt.show()

        m2u = Music21Util()
        # ms = part.measures(20, 100)
        # m2u.plot_piano_roll(ms, s=10, e=30)
        # m2u.plot_piano_roll(scr, s=10, e=15)
        m2u.plot_piano_roll(part, s=20, e=40)
        # mic(type(ms), vars(ms), dir(ms))
        # mic(ms.measures(26, 30))
        # mic(mes, nums)
        # for mes in ms.measures(0, None):
        #     mic(mes)
    # test_piano_roll()

    def check_show_title():
        fnm = music_util.get_my_example_songs('Merry Go Round of Life', fmt='MXL')
        mic(fnm)
        scr = m21.converter.parse(fnm)
        mic(scr)
        # mic(len(dir(scr)))
        # mic(vars_(scr, include_private=False))
        meta = scr.metadata
        # mic(meta, vars(meta), vars_(meta))
        mic(meta.title, meta.composer)
        part_ch2 = scr.parts[1]
        mic(part_ch2, part_ch2.partName, part_ch2.metadata)
        # mic(vars(part_ch2), vars_(part_ch2))
        mic(part_ch2.activeSite.metadata.title)
    # check_show_title()

    def check_tuplet_duration_creation():
        """
        Make sure too small durations in a tuplet can still be shown
        """
        lst_note = [Note(pitch=m21.pitch.Pitch(midi=59), duration=m21.duration.Duration(quarterLength=1))]
        lst_tup = []
        for i in range(19):
            dur = m21.duration.Tuplet(numberNotesActual=19, numberNotesNormal=3)
            note = m21.note.Note()
            note.duration.appendTuplet(dur)
            mic(note.duration.quarterLength)
            lst_tup.append(note)
        lst_note.append(tuple(lst_tup))
        lst_note.append(Note(pitch=m21.pitch.Pitch(midi=80), duration=m21.duration.Duration(quarterLength=2)))
        lst_note = [note2note_cleaned(n, for_output=True) for n in lst_note]
        mic(lst_note)
        lst_note = list(flatten_notes(lst_note))

        bar = m21.stream.Measure()
        bar.append(lst_note)
        for n in bar[Note]:
            mic(n, n.duration.quarterLength, n.offset)
        # bar.show()
    check_tuplet_duration_creation()
