"""
Since Sun. Jan. 30th, an updated module for music/melody extraction, with a duration-quantized approach

See `melody_extractor` for the old version.
"""

import math
import datetime
import itertools
from os.path import join as os_join
from copy import deepcopy
from typing import List, Tuple, Dict, Iterable, Union, Any
from fractions import Fraction
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
import music21 as m21

from stefutil import *
from musicnlp.util import *
from musicnlp.util.music_lib import *
from musicnlp.vocab import COMMON_TEMPOS, COMMON_TIME_SIGS, is_common_tempo, is_common_time_sig, MusicVocabulary
from musicnlp.preprocess.warning_logger import WarnLog
from musicnlp.preprocess.key_finder import KeyFinder


__all__ = ['MusicExtractorOutput', 'MusicExtractor']


@dataclass
class BarInfo:
    bars: Union[Tuple[Measure], List[Measure]]
    time_sig: TimeSignature
    tempo: MetronomeMark


ExtractedNotes = List[List[ExtNote]]


@dataclass
class MusicExtractorOutput:
    score: ScoreExt = None
    song_path: str = None
    title: str = None
    duration: int = None
    warnings: List[Dict[str, Any]] = None
    keys: Dict[str, float] = None


class MusicExtractor:
    """
    Extract melody and potentially chords from MXL music scores => An 1D polyphonic representation
    """
    def __init__(
            self, precision: int = 5, mode: str = 'melody', with_pitch_step: bool = False,
            warn_logger: Union[WarnLog, bool] = None,
            greedy_tuplet_pitch_threshold: int = 3**9,
            verbose: Union[bool, str] = True,
            epsilon: float = eps
    ):
        """
        :param precision: Bar duration quantization, see `melody_extractor.MxlMelodyExtractor`
        :param mode: Extraction mode, one of [`melody`, `full`]
            `melody`: Only melody is extracted
            `full`: Melody and Bass as 2 separate channels extracted
        :param with_pitch_step: If true, the scale degree of each note is included in the pitch token
        :param warn_logger: A logger for storing warnings
            If True, a logger is instantiated
        :param greedy_tuplet_pitch_threshold: If #possible note cartesian product in the tuplet > threshold,
                only the note with the highest pitch in chords in tuplets is kept
            Set to a small number to speedup processing, e.g. 1 for always keeping the highest notes
                Experimental, not sure if the tokens extracted would be different
            It's not advised to pass too large numbers, as possible Chord notes per tuplet may get prohibitively large
                due to transcription quality - See `expand_bar`
        :param verbose: If true, extraction process including warnings is logged
            If `single`, only begin and end of extraction is logged
        :param epsilon: For float- & Fraction-related equality comparison

        .. note:: Prior logging warning messages are removed after new encode call, see `Warning.end_tracking`
        """
        self.prec = precision
        ca.check_mismatch('Music Extraction Mode', mode, ['melody', 'full'])
        self.mode = mode
        self.pc = PrecisionChecker(precision=self.prec)

        self.logger = get_logger('Music Extraction')
        assert isinstance(verbose, bool) or verbose == 'single', f'{pl.i("verbose")} must be bool or {pl.i("single")}'
        if warn_logger:
            self.warn_logger = warn_logger if isinstance(warn_logger, WarnLog) else WarnLog(verbose=verbose is True)
        else:
            self.warn_logger = None
        self.greedy_tuplet_pitch_threshold = greedy_tuplet_pitch_threshold
        self.verbose = verbose
        self.eps = epsilon

        self.vocab = MusicVocabulary(precision=precision, pitch_kind='step' if with_pitch_step else 'midi')

        self.meta = dict(
            mode=mode, precision=precision, with_pitch_step=with_pitch_step,
            greedy_tuplet_pitch_threshold=greedy_tuplet_pitch_threshold
        )

    @staticmethod
    def meta2fnm_meta(d: Dict = None, short: bool = True) -> str:
        keys = ('md', 'prec', 'th') if short else ('mode', 'precision', 'threshold')
        vals = (d['mode'][0] if short else d['mode']), d['precision'], d['greedy_tuplet_pitch_threshold']
        return pl.pa(dict(zip(keys, vals)))

    def log_warn(self, log_d: Dict = None, **kwargs):
        d = log_d or kwargs
        if self.warn_logger is not None:
            self.warn_logger: WarnLog
            self.warn_logger.update(d)

    def dur_within_prec(self, dur: Union[float, Fraction]) -> bool:
        return is_int(dur / 4 / (2**-self.prec))

    def it_bars(self, scr: Score) -> Iterable[BarInfo]:
        """
        Unroll a score by time, with the time signatures of each bar
        """
        parts = list(scr.parts)
        ignore = [is_drum_track(p_) for p_ in parts]

        time_sig, tempo = None, None
        for idx, bars in enumerate(zip(*[list(p[Measure]) for p in parts])):  # Bars for all tracks across time
            # Still enumerate the to-be-ignored tracks for getting time signature and tempo
            assert list_is_same_elms([b.number for b in bars]), 'Bar numbers should be the same'

            # Update time signature
            tss = [b[TimeSignature] for b in bars]
            if idx == 0 or any(tss):  # 1st bar must have time signature defined
                # some parts may contain no time signature if the file went through Logic Pro mid=>xml conversion
                tss = [list(t) for t in tss if t]
                assert all(len(t) == 1 for t in tss)
                tss = [t[0] for t in tss]
                assert list_is_same_elms([(ts.numerator, ts.denominator) for ts in tss])
                time_sig = tss[0]

            tempos = [list(b[MetronomeMark]) for b in bars]
            # observed tempo with number `None`... in such case, ignore
            has_tempo = any(tempos) and any(any(t.number for t in ts) for ts in tempos)
            if has_tempo:
                tempos = [t for t in tempos if len(t) != 0]
                # When multiple tempos, take the mean
                tempos = [MetronomeMark(number=np.mean([t.number for t in ts])) for ts in tempos]
                bpms = [t.number for t in tempos]
                assert list_is_same_elms(bpms)
                tempo = MetronomeMark(number=bpms[0])
            elif idx == 0:
                self.log_warn(warn_name=WarnLog.MissTempo)
                tempo = MetronomeMark(number=120)  # set as default
            yield BarInfo(bars=[b for ignore, b in zip(ignore, bars) if not ignore], time_sig=time_sig, tempo=tempo)

    @staticmethod
    def chord2notes(c: Chord) -> List[Union[Rest, Note]]:
        notes = list(c.notes)
        for i_ in range(len(notes)):  # Offsets for notes in chords are 0, restore them
            notes[i_].offset = c.offset
        return notes

    def expand_bar(
            self, bar: Union[Measure, Voice], time_sig: TimeSignature, keep_chord=False, number=None
    ) -> List[ExtNote]:
        """
        Expand elements in a bar into individual notes, no order is enforced

        :param bar: A music21 measure to expand
        :param time_sig: Time signature of the bar
        :param keep_chord: If true, `Chord`s are not expanded
        :param number: For passing bar number recursively to Voice

        .. note:: Triplets (potentially any n-plets) are grouped; `Voice`s are expanded
        """
        lst = []
        it = iter(bar)
        elm = next(it, None)
        while elm is not None:
            elm: m21.stream.Stream
            # this is the bottleneck; just care about duration; Explicitly ignore voice
            full_nm = not isinstance(elm, Voice) and getattr(elm.duration, 'fullName', None)
            if full_nm and tuplet_postfix in full_nm:
                tup_str, n_tup = fullname2tuplet_meta(full_nm)
                n_ignored, tup_ignored = 0, False  # for sanity check, see below (`LowTupDur`)

                elms_tup: List[Union[Rest, Note, Chord]] = [elm]
                elm_ = next(it, None)
                while elm_ is not None:
                    # For poor transcription quality, skip over non-note elements in the middle
                    if isinstance(elm_, (m21.clef.Clef, MetronomeMark, m21.bar.Barline)):
                        elm_ = next(it, None)
                    elif tup_str in elm_.duration.fullName:  # Look for all elements of the same `n_tup`
                        elms_tup.append(elm_)
                        elm_ = next(it, None)  # Peeked 1 ahead
                    else:  # Finished looking for all tuplets
                        break

                def get_filled_ranges():  # cache
                    if not hasattr(get_filled_ranges, 'filled_ranges'):
                        get_filled_ranges.filled_ranges = [
                            (serialize_frac(get_offset(n)), serialize_frac(get_end_qlen(n))) for n in elms_tup
                        ]
                    return get_filled_ranges.filled_ranges
                if notes_overlapping(elms_tup):
                    self.log_warn(warn_name=WarnLog.TupNoteOvlIn, bar_num=number, filled_ranges=get_filled_ranges())
                if notes_have_gap(elms_tup, enforce_no_overlap=False):
                    ts = (time_sig.numerator, time_sig.denominator)
                    self.log_warn(
                        warn_name=WarnLog.TupNoteGap, bar_num=number, time_sig=ts, filled_ranges=get_filled_ranges()
                    )

                # Consecutive tuplet notes => (potentially multiple) groups
                it_tup = iter(elms_tup)
                e_tup = next(it_tup, None)
                dur: Union[Fraction, float] = 0
                idx, idx_next_strt, idx_last = 0, 0, len(elms_tup)-1
                n_tup_curr = 0
                tup_added = False
                idx_tup_strt = len(lst)
                is_single_tup = False  # Edge case, see below

                # MIDI & MuseScore transcription quality, e.g. A triplet may not contain 3 notes
                while e_tup is not None:
                    dur += e_tup.duration.quarterLength
                    n_tup_curr += 1
                    # Enforce a tuplet must have at least `n_tup` notes
                    # Duration for a (normal) tuplet must be multiples of 8th note; Heuristic for end of tuplet group
                    if n_tup_curr >= n_tup and is_8th(dur):
                        lst.append(tuple(elms_tup[idx_next_strt:idx+1]))
                        tup_added = True

                        # Prep for next tuplet
                        idx_next_strt = idx+1
                        n_tup_curr = 0
                        dur = 0

                    if idx == idx_last:  # Processed til last element, make sure all tuplet notes are added
                        if len(elms_tup) == 1:  # Poor processing, edge case
                            note = elms_tup[0]  # As if single note with weird duration
                            if (not keep_chord) and isinstance(note, Chord):
                                note = max(note, key=lambda n: note2pitch(n))
                            lst.append(note)
                            tup_added, is_single_tup = True, True
                            e_tup = None
                            continue  # Break out since done processing this tuplet group

                        if is_8th(dur) and n_tup_curr < n_tup:  # Last tuplet group not enough elements
                            if tup_added:  # Join the prior tuplet group if possible
                                lst[-1] = lst[-1] + tuple(elms_tup[idx_next_strt:])
                            else:  # Number of tuplet notes not enough, create new tuplet anyway
                                tup_added = True
                                lst.append(tuple(elms_tup[idx_next_strt:]))
                        else:  # Resort to the erroneous case only til the end
                            # Must be that duration of all triplet elements don't sum up to an 8th note in quarterLength
                            assert not is_8th(dur)  # Add such tuplet notes anyway, set a valid quantized length
                            warn_nm = WarnLog.InvTupDur
                            offsets, durs = notes2offset_duration(elms_tup[idx_next_strt:])
                            curr_ignored = False
                            if not self.dur_within_prec(dur):  # Enforce it by changing the durations
                                warn_nm = WarnLog.InvTupDurSv

                                def round_by_factor(num, fact):
                                    return round(num / fact) * fact
                                # Round to quantization duration; Crop by bar max duration
                                dur = min(
                                    round_by_factor(dur, 2**-self.prec), time_sig.numerator/time_sig.denominator*4
                                )
                                n_tup_last = len(elms_tup[idx_next_strt:])
                                if dur > 0:
                                    dur_ea = dur / n_tup_last
                                    strt = elms_tup[idx_next_strt].offset
                                    for i in range(idx_next_strt, idx_next_strt+n_tup_last):
                                        elms_tup[i].offset = strt
                                        dur_ = m21.duration.Duration(quarterLength=dur_ea)
                                        elms_tup[i].duration = dur_
                                        if isinstance(elms_tup[i], Chord):
                                            cd = elms_tup[i]
                                            for i_n in range(len(cd.notes)):  # Broadcast duration to notes inside
                                                elms_tup[i].notes[i_n].duration = dur_
                                        strt += dur_ea
                                else:  # total duration for the group is way too small, ignore
                                    n_ignored += n_tup_last
                                    tup_ignored = curr_ignored = True
                                    ts = (time_sig.numerator, time_sig.denominator)
                                    self.log_warn(
                                        warn_name=WarnLog.LowTupDur, bar_num=number,
                                        time_sig=ts, precision=self.prec, filled_ranges=get_filled_ranges()
                                    )
                            if not curr_ignored:
                                lst.append(tuple(elms_tup[idx_next_strt:]))
                                tup_added = True
                            self.log_warn(warn_name=warn_nm, bar_num=number, offsets=offsets, durations=durs)
                    idx += 1
                    e_tup = next(it_tup, None)
                # All triple notes with the same `n_tup` are added
                assert tup_added or tup_ignored
                if not is_single_tup:  # sanity check all notes are processed
                    assert sum(len(tup) for tup in lst[idx_tup_strt:]) + n_ignored == len(elms_tup)

                    for tup in lst[idx_tup_strt:]:
                        ln = len(tup)
                        # assert ln != 2 and ln != 4  # sanity check TODO: include
                        if ln != n_tup:
                            self.log_warn(warn_name=WarnLog.InvTupSz, bar_num=number, n_expect=n_tup, n_got=ln)

                    # Enforce no overlap in each triplet group
                    for idx_tup, tup in enumerate(lst[idx_tup_strt:], start=idx_tup_strt):
                        tup: tuple[Union[Note, Rest]]
                        if notes_overlapping(tup):
                            offsets, durs = notes2offset_duration(tup)
                            self.log_warn(
                                warn_name=WarnLog.TupNoteOvlOut, bar_num=number, offsets=offsets, durations=durs
                            )
                            # TODO: how about just remove this group?
                            total_dur: Union[float, Fraction] = sum(n.duration.quarterLength for n in tup)
                            dur_16th = 4 / 16  # duration in quarter length
                            # Trust the duration more than the offset, and unroll backwards to fix the offset
                            # As long as total duration is still multiple of 16th note, make the offset work
                            multiplier: Union[float, Fraction] = total_dur / dur_16th

                            assert float_is_int(multiplier, eps=self.eps) if isinstance(multiplier, float) \
                                else multiplier.denominator == 1
                            note1st = note2note_cleaned(tup[0])
                            offset = note1st.offset + note1st.duration.quarterLength
                            fixed_tup = [note1st]
                            for n in tup[1:]:
                                n = note2note_cleaned(n)
                                n.offset = offset
                                fixed_tup.append(n)
                                offset += n.duration.quarterLength
                            assert not notes_overlapping(fixed_tup)  # sanity check
                            lst[idx_tup] = tuple(fixed_tup)  # Override the original tuplet
                    for tup in lst[idx_tup_strt:]:
                        n_rest = sum(isinstance(n, Rest) for n in tup)
                        if n_rest != 0:
                            self.log_warn(warn_name=WarnLog.RestInTup, bar_num=number, n_rest=n_rest, n_note=len(tup))

                    if not keep_chord:
                        tups_new = []
                        has_chord = False
                        for i in range(idx_tup_strt, len(lst)):  # Ensure all tuplet groups contain no Chord
                            tup = lst[i]
                            # Bad transcription quality => Keep all possible tuplet combinations
                            # Try to, but all possible search space is huge as we recurse, see `get_notes_out`
                            # Expect to be the same
                            if any(isinstance(n, Chord) for n in tup):
                                has_chord = True
                                opns = [MusicExtractor.chord2notes(n) if isinstance(n, Chord) else (n,) for n in tup]
                                # Adding all possible tuplet notes may be the bottleneck during extraction
                                n_opns = [len(n) for n in opns if n]
                                if math.prod(n_opns) > self.greedy_tuplet_pitch_threshold:
                                    # Too much possible cartesian products for later processing to handle
                                    # as it involves sorting
                                    # Cap at a tuplet of 9 consecutive 3-note Chords, beyond this number,
                                    # just treat the bar as wicked
                                    self.log_warn(
                                        warn_name=WarnLog.ExcecTupNote, bar_num=number, note_choices=n_opns,
                                        threshold=self.greedy_tuplet_pitch_threshold
                                    )
                                    notes_max_pitch = tuple([max(notes, key=note2pitch) for notes in opns])
                                    tups_new.append(notes_max_pitch)
                                else:
                                    tups_new.extend(list(itertools.product(*opns)))
                            else:  # keep the tuplet group
                                tups_new.append(tup)
                        if has_chord:  # Replace prior triplet groups
                            lst = lst[:idx_tup_strt] + tups_new
                for idx_tup, tup in enumerate(lst[idx_tup_strt:], start=idx_tup_strt):
                    if isinstance(tup, tuple) and len(tup) == 1:
                        lst[idx_tup] = tup[0]  # consider as single note if just 1 element
                elm = elm_
                continue  # Skip `next` for peeked 1 step ahead
            elif isinstance(elm, (Note, Rest)):
                lst.append(elm)
            elif isinstance(elm, Chord):
                if keep_chord:
                    lst.append(elm)
                else:
                    notes = list(elm.notes)
                    for n in notes:  # TODO: for efficiency, keep just the top note?
                        n.offset += elm.offset  # Shift offset in the scope of bar
                    lst.extend(notes)
            else:
                if not isinstance(elm, (  # Ensure all relevant types are considered
                    TimeSignature, MetronomeMark, Voice,
                    m21.layout.LayoutBase, m21.clef.Clef, m21.key.KeySignature, m21.bar.Barline,
                    m21.expressions.TextExpression, m21.repeat.Fine
                )):
                    raise TypeError(f"Unexpected element {pl.i(elm)} w/ type {pl.i(type(elm))}")
            elm = next(it, None)
        assert is_notes_pos_duration(lst)
        if bar.hasVoices():  # Join all voices to notes
            lst.extend(chain_its(self.expand_bar(v, time_sig, number=number) for v in bar.voices))
        if not keep_chord:  # sanity check
            assert all(
                all(not isinstance(n_, Chord) for n_ in n) if isinstance(n, tuple) else (not isinstance(n, Chord))
                for n in lst
            )
        return lst

    @staticmethod
    def sort_groups(groups, reverse: bool = False):
        for offset, ns in groups.items():  # sort by pitch then by duration, in-place for speed
            # Create shallow copy of list so that no aliasing in list append & removal in `get_notes_out`
            groups[offset] = sorted(ns, key=lambda nt: (note2pitch(nt), note2dur(nt)), reverse=reverse)

    @staticmethod
    def _time_same(a: Union[Note, Rest], b: Union[Note, Rest]) -> bool:
        return a.offset == b.offset and a.duration.quarterLength == b.duration.quarterLength

    @staticmethod
    def _pitch_same(a: Union[Note, Rest], b: Union[Note, Rest]) -> bool:
        return a.pitch.midi == b.pitch.midi

    @staticmethod
    def _ext_notes_eq(nt1: ExtNote, nt2: ExtNote):
        """
        For filtering out notes that appear in melody & in bass
        """
        if type(nt1) != type(nt2):
            return False
        else:
            assert isinstance(nt1, (Rest, Note, tuple))
            if isinstance(nt1, Rest):
                return MusicExtractor._time_same(nt1, nt2)
            elif isinstance(nt1, Note):
                return MusicExtractor._time_same(nt1, nt2) and MusicExtractor._pitch_same(nt1, nt2)
            else:
                return len(nt1) == len(nt2) and all(
                    MusicExtractor._ext_notes_eq(n1_, n2_) for n1_, n2_ in zip(nt1, nt2)
                )

    @staticmethod
    def _deep_copy_note(note: ExtNote):
        if isinstance(note, tuple):
            return tuple([MusicExtractor._deep_copy_note(n) for n in note])
        else:
            n = deepcopy(note)
            n.offset = note.offset
            return n

    def extract_notes(
            self, lst_bar_info: List[BarInfo], time_sigs: List[TimeSignature]
    ) -> Dict[str, ExtractedNotes]:
        lst_melody: ExtractedNotes = []
        lst_bass: ExtractedNotes = []
        i_bar_strt = lst_bar_info[0].bars[0].number  # Get number of 1st bar
        for i_bar, bi in enumerate(lst_bar_info):
            bars, time_sig, tempo = bi.bars, bi.time_sig, bi.tempo
            number = bars[0].number - i_bar_strt  # Enforce bar number 0-indexing
            assert number == i_bar  # sanity check
            # For now, with melody only or melody + bass, always convert Chords into single notes TODO
            all_notes = sum((self.expand_bar(b, time_sig, keep_chord=False, number=number) for b in bars), [])

            groups_melody: Dict[float, List[ExtNote]] = defaultdict(list)  # Group notes by starting location
            for n in all_notes:
                groups_melody[get_offset(n)].append(n)
            # if number == 0:
            #     mic(groups_melody)
            #     for offset, notes in groups_melody.items():
            #         mic(offset)
            #         debug_pprint_lst_notes(notes)
                # raise NotImplementedError
            # In rare occasions, extracted melody notes are empty, while bass notes are not
            #   e.g. 0-th bar of `LMD::068052`, where a tuplet group is dropped,
            #       due to a note w/ higher pitch & tiny duration occurring during the tuplet group
            #       which is then removed during quantization
            MusicExtractor.sort_groups(groups_melody, reverse=False)
            groups_melody = self._fix_edge_case(groups_melody, number, time_sig)
            groups_bass = None
            if self.mode == 'full':
                # make deep copy, since melody extraction in `get_notes_out` modify `groups`
                # filter out all Rests to try to get notes TODO: update Bar gap warning?
                groups_bass = {
                    k: [MusicExtractor._deep_copy_note(n) for n in v if not is_rest(n)]
                    for k, v in groups_melody.items()
                }
                # if number == 674:
                #     mic(groups_bass)
                #     for offset, notes in groups_bass.items():
                #         mic(offset)
                #         debug_pprint_lst_notes(notes)
                # so that accessing last element gives the smallest pitch, see `get_notes_out`
                MusicExtractor.sort_groups(groups_bass, reverse=True)

            def _local_post_process(notes_):
                self.warn_notes_duration(notes_, time_sig, number)
                self.warn_notes_overlap(notes_, number)
                return [note2note_cleaned(nt) for nt in notes_]

            def _get_notes(groups_, keep: str = 'high'):
                with RecurseLimit(2 ** 14):
                    return self.get_notes_out(groups_, number, keep=keep)
            notes_melody = _get_notes(groups_melody, 'high')
            # if number == 0:
            #     mic(notes_melody)
            #     debug_pprint_lst_notes(notes_melody)
            #     raise NotImplementedError
            if self.mode == 'full':
                _notes_bass = _get_notes(groups_bass, 'low')

                notes_bass, removed = [], False
                for nb in _notes_bass:
                    # only keep the notes that are unique to bass
                    if not any(MusicExtractor._ext_notes_eq(nb, nm) for nm in notes_melody):
                        notes_bass.append(nb)
                        removed = True
                    elif self.verbose:
                        self.logger.info(
                            f'Skipping {pl.i("bass")} note at bar#{number}: {pl.i(nb)} for already in melody')
                if removed:
                    # skip unfilled range
                    notes_bass = fill_with_rest(notes_bass, duration=time_sig2bar_dur(time_sig), fill_start=True)[0]
                # if number == 674:
                #     mic(notes_bass)
                #     debug_pprint_lst_notes(notes_bass)
                lst_bass.append(_local_post_process(notes_bass))
            # For poor transcription quality, postpone `is_valid_bar_notes` *assertion* until after quantization,
            # since empirically observe notes don't sum to bar duration,
            #   e.g. tiny-duration notes shifts all subsequent notes
            #     n: <music21.note.Rest inexpressible>
            #     n.fullName: 'Inexpressible Rest'
            #     n.offset: 2.0
            #     n.duration.quarterLength: Fraction(1, 480)
            lst_melody.append(_local_post_process(notes_melody))
        d = dict(melody=self._post_process(lst_melody, time_sigs))
        if self.mode == 'full':
            d['bass'] = self._post_process(lst_bass, time_sigs)
        return d

    @staticmethod
    def _fix_rest_too_long(groups, offset, wrong_end_time):
        if offset in groups:
            _notes_out = []
            for _n in groups[offset]:  # starts at offset 4
                if isinstance(_n, Rest) and get_end_qlen(_n) == wrong_end_time:  # 4 qlen more than it should
                    continue  # ignore; then if no notes, will fill with rest with subsequent logic
                _notes_out.append(_n)
            groups[offset] = _notes_out

    @staticmethod
    def _fix_truncate_note(groups, ts_tup: Tuple[int, int], offset, wrong_end_time):
        # may happen if a chord starts in 2.125, and all its notes have duration 1 in quarter length,
        # 1/8 more than it should
        # TODO: only fix the case where Chords are broken down into notes
        if offset in groups:
            notes_out = []
            dur_bar = time_sig2bar_dur(ts_tup)
            for n in groups[offset]:
                if isinstance(n, Note) and get_end_qlen(n) == wrong_end_time:
                    assert offset == n.offset  # sanity check
                    n.duration = Duration(quarterLength=dur_bar - offset)
                notes_out.append(n)
            groups[offset] = notes_out

    @staticmethod
    def _fix_long_tuplets(groups, ts_tup, offset, wrong_end_time):
        if offset in groups:
            notes_out = []
            dur_bar = time_sig2bar_dur(ts_tup)
            for n in groups[offset]:
                if isinstance(n, tuple) and get_end_qlen(n) == wrong_end_time:
                    n = note2note_cleaned(n, q_len=dur_bar - offset)  # Keep, but normalize
                notes_out.append(n)
            groups[offset] = notes_out

    def _fix_edge_case(self, groups, number, time_sig):
        """
        the original file is broken in that note durations don't align with duration by time signature
        """
        ts_tup = (time_sig.numerator, time_sig.denominator)
        if ts_tup in [(8, 4), (4, 2), (2, 1)] and \
                number in [9, 17, 19, 24, 33, 38, 43, 47, 52, 53, 60, 62, 84, 87, 188, 201] and \
                (4.0 in groups or 6.0 in groups):
            # for [
            #   `LMD::027213`, `LMD::`050735`, `LMD::054246`, `LMD::069877`, `LMD::108367`,
            #   `LMD::116976`, `LMD::119887`, `LMD::123389`, `LMD::128869`, `LMD::137904`,
            #   `LMD::140453`,`LMD::142327`,`LMD::160646`, `LMD::161475`, `LMD::163655`
            # ]
            for offset in [4.0, 6.0]:
                MusicExtractor._fix_rest_too_long(groups, offset, 12.0)
        elif ts_tup == (2, 4) and number == 6 and 2.0 in groups:
            MusicExtractor._fix_rest_too_long(groups, 2.0, 4.0)  # for `LMD::034249`
        elif ts_tup == (1, 8):
            # for `LMD::051562`, `LMD::119192`
            if number in [9, 40, 60, 71, 88, 102] and all(o in groups for o in [0.5, 4.0, 8.0]):
                for offset, wrong_time in [(0.0, 4.0), (0.5, 12.0), (4.0, 8.0), (4.0, 12.0), (8.0, 12.0)]:
                    MusicExtractor._fix_rest_too_long(groups, offset, wrong_time)
            elif number in [26, 57, 88] and all(o in groups for o in [0.0, 4.0, 8.0]):
                for offset, wrong_time in [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)]:
                    MusicExtractor._fix_rest_too_long(groups, offset, wrong_time)
        elif ts_tup == (5, 2) and number in [5, 28]:
            MusicExtractor._fix_rest_too_long(groups, 6.0, 16.0)  # for `LMD::109166`
        elif ts_tup == (4, 4):
            if number == 1:
                MusicExtractor._fix_long_tuplets(groups, ts_tup, 0.0, Fraction(33, 8))  # for `LMD::116496`
            elif number == 12:
                MusicExtractor._fix_truncate_note(groups, ts_tup, 3.875, 4.875)  # for `LMD::090283`
            elif number == 27:
                MusicExtractor._fix_truncate_note(groups, ts_tup, 3.25, 4.25)
            elif number in (42, 90, 97, 621):
                # for [
                #   `MAESTRO::`Frédéric Chopin - Variations And Fugue In E-flat Major, Op. 35, "eroica"`,
                #   `LMD::074940`, `LMD::084360`, `LMD::096500`
                # ]
                MusicExtractor._fix_long_tuplets(groups, ts_tup, 2.0, Fraction(33, 8))
            elif number == 46 and 4.0 in groups:  # for `LMD::086800`
                _notes_out = []
                for _n in groups[4.0]:
                    e = get_end_qlen(_n)
                    if isinstance(_n, Rest) and \
                            (math.isclose(e, 4.110416666666667, abs_tol=self.eps) or e == 4.125):
                        continue  # ignore
                    _notes_out.append(_n)
                groups[4.0] = _notes_out
            elif number == 56:
                MusicExtractor._fix_long_tuplets(groups, ts_tup, 3.0, Fraction(33, 8))  # for `LMD::098334`
            elif number == 65:
                MusicExtractor._fix_truncate_note(groups, ts_tup, 3.25, 4.25)  # for `LMD::173000`
            elif number == 108:
                MusicExtractor._fix_truncate_note(groups, ts_tup, 3.75, 4.75)  # for `LMD::173000`
        elif ts_tup == (3, 4):
            if number == 22:
                MusicExtractor._fix_truncate_note(groups, ts_tup, 2.125, 3.125)
            elif number == 48:
                MusicExtractor._fix_truncate_note(groups, ts_tup, 2.375, 3.375)  # for `LMD::104680`
            elif number == 85:
                MusicExtractor._fix_truncate_note(groups, ts_tup, 2.625, 3.625)  # for `LMD::104680`
            elif number == 91:
                MusicExtractor._fix_truncate_note(groups, ts_tup, 2.875, 3.875)  # for `LMD::060134`
            elif number == 96:
                MusicExtractor._fix_truncate_note(groups, ts_tup, 2.5, 3.5)  # for `LMD::161651`
            elif number == 126:
                MusicExtractor._fix_truncate_note(groups, ts_tup, 2.75, 3.75)  # for `LMD::051872`
            elif number == 326:
                MusicExtractor._fix_long_tuplets(groups, ts_tup, 1.0, 3.375)  # for `LMD::061641`
            elif number == 674:
                MusicExtractor._fix_long_tuplets(groups, ts_tup, 0.0, Fraction(4, 1))  # for `LMD::107205`
        return groups

    def warn_notes_duration(self, notes: List[ExtNote], time_sig: TimeSignature, number: int):
        if not math.isclose(get_notes_duration(notes), time_sig2bar_dur(time_sig), abs_tol=self.eps):
            offsets, durs = notes2offset_duration(notes)
            self.log_warn(  # can be due to removing lower-pitched tuplets
                warn_name=WarnLog.InvBarDur, bar_num=number, offsets=offsets, durations=durs, time_sig=time_sig
            )

    def warn_notes_overlap(self, notes: List[ExtNote], number: int):
        if notes_overlapping(notes):
            # The source of overlapping should be inside tuplet
            assert not non_tuplet_notes_overlapping(notes)
            for tup in notes:
                if isinstance(tup, tuple) and notes_overlapping(tup):
                    offsets, durs = notes2offset_duration(tup)
                    self.log_warn(warn_name=WarnLog.TupNoteOvlOut, bar_num=number, offsets=offsets, durations=durs)

    def get_notes_out(
            self, groups: Dict[float, List[ExtNote]], number: int, keep: str = 'high', pre_sort: bool = False
    ) -> List[ExtNote]:
        """
        :param groups: Notes grouped by offset, see `__call__`
        :param number: current bar # in processing
        :param keep: Which note pitch extreme to keep
            If `high`, keep notes with the highest pitch
            If `low`, keep notes with the lowest pitch
        :param pre_sort: If true, notes are sorted before extraction
            Intended for internal recursion when note durations are truncated
        """
        # if number == 0:
        #     mic('in new get notes out', groups, keep)
        #     for offset, notes in groups.items():
        #         mic(offset)
        #         debug_pprint_lst_notes(notes)
        is_high = keep == 'high'
        if pre_sort:
            MusicExtractor.sort_groups(groups, reverse=not is_high)
        pre_sort = False
        ns_out = []
        last_end = 0  # Last added note, end time in quarter length
        for offset in sorted(groups.keys()):  # Pass through notes in order
            notes_ = groups[offset]
            if len(notes_) == 0:  # As a result of removing triplets
                del groups[offset]
                continue
            nt = notes_[-1]  # Note with the highest pitch if `high`, the lowest if `low`
            # if number == 0:
            #     mic(offset, ns_out, nt)
            nt_end = get_end_qlen(nt)
            if last_end - offset > self.eps:
                # Current note starts before the last added note ends
                # Tuplet notes not normalized at this point, remain faithful to the weighted average pitch
                note_last = ns_out[-1]
                pch_last, pch_curr = note2pitch(note_last), note2pitch(nt)
                later_note_better_pitch = pch_curr > pch_last if is_high else pch_curr < pch_last
                if later_note_better_pitch:  # Truncate last added note
                    if isinstance(note_last, tuple):  # tuplet being truncated => Remove entirely, start over
                        # The triplet must've been the last note added, and it's joint offset is known
                        del groups[get_offset(note_last)][-1]
                        self.log_warn(warn_name=WarnLog.HighPchOvlTup, bar_num=number)
                        return self.get_notes_out(groups, number, keep=keep, pre_sort=pre_sort)
                    else:
                        self.log_warn(warn_name=WarnLog.HighPchOvl, bar_num=number)

                        nt_ = nt[0] if isinstance(nt, tuple) else nt
                        # Resulting duration usually non-0, for offset grouping
                        note_last.duration = dur_last = Duration(quarterLength=nt_.offset - note_last.offset)
                        # Need notes sorting if ever start from scratch, cos shorter duration
                        pre_sort = True
                        assert dur_last.quarterLength >= 0
                        # If it's 0, it's cos a **truncated** note was appended, as makeup
                        if dur_last.quarterLength == 0:  # TODO: verify
                            note_2_delete = ns_out.pop()
                            assert note_2_delete.offset == offset
                            assert groups[offset][-1] == note_2_delete
                            del groups[offset][-1]
                            self.log_warn(warn_name=WarnLog.LowPchMakeupRmv, bar_num=number)
                    ns_out.append(nt)
                    last_end = nt_end
                # Current note to add has lower pitch, but ends later in time than the last note added
                # Truncate current note, add back into group based on new start time, recompute
                elif (not later_note_better_pitch) and (nt_end - last_end) > self.eps:  # do so for bass too
                    if not isinstance(nt, tuple):
                        # Move the truncated note to later group, restart
                        del groups[offset][-1]
                        nt_ = note2note_cleaned(nt)
                        nt_.offset = last_end
                        nt_.duration = d = Duration(quarterLength=nt_end - last_end)
                        assert d.quarterLength > 0
                        last_end_closest = min(groups.keys(), key=lambda x: abs(x - last_end))
                        # since Fractions and float that are real close, are not equal (==)
                        if abs(last_end - last_end_closest) < self.eps:
                            last_end = last_end_closest
                        if last_end in groups:
                            groups[last_end].append(nt_)
                        else:
                            groups[last_end] = [nt_]
                        MusicExtractor.sort_groups(groups, reverse=not is_high)  # sort in reverse for bass
                        self.log_warn(warn_name=WarnLog.LowPchMakeup, bar_num=number)
                        return self.get_notes_out(groups, number, keep=keep, pre_sort=pre_sort)
                    # Skip adding tuplets, this potentially leads to gaps in extraction output
                # Otherwise, skip if later note is lower in pitch and is covered by the prior note duration
            else:
                ns_out.append(nt)
                last_end = nt_end
        return ns_out

    def _post_process(self, lst_notes, time_sigs: List[TimeSignature]):
        # Enforce quantization
        for i_bar, (notes, time_sig) in enumerate(zip(lst_notes, time_sigs)):
            dur = time_sig2bar_dur(time_sig)
            if not self.pc.notes_within_prec(notes):
                lst_notes[i_bar] = self.notes2quantized_notes(notes, time_sig, number=i_bar)
                assert self.pc.notes_within_prec(lst_notes[i_bar])  # Sanity check implementations
                offsets, durs = notes2offset_duration(notes)
                self.log_warn(warn_name=WarnLog.NoteNotQuant, bar_num=i_bar, offsets=offsets, durations=durs)
            elif notes_have_gap(notes, duration=dur):
                lst_notes[i_bar], unfilled_ranges = fill_with_rest(notes, duration=dur)
                self.log_warn(
                    warn_name=WarnLog.BarNoteGap, bar_num=i_bar,
                    time_sig=(time_sig.numerator, time_sig.denominator),
                    precision=self.prec, unfilled_ranges=unfilled_ranges
                )
        # Now, triplets fixed to equal duration by `clean_quantized_tuplets`
        lst_notes = [self.clean_quantized_tuplets(notes, num_bar=i) for i, notes in enumerate(lst_notes)]

        for notes in lst_notes:
            for n in notes:
                if not isinstance(n, tuple):  # ignore tuplet durations as `consolidate` doesn't consider tuplets
                    # Merges complex durations into one for MXL output
                    n.duration.consolidate()
        for i_bar, (notes, time_sig) in enumerate(zip(lst_notes, time_sigs)):  # Final check before output
            # for `LMD::125135`, edge case, see `notes2quantized_notes`
            if self.prec == 5 and (time_sig.numerator, time_sig.denominator) == (21, 64) and i_bar == 51:
                check_dur = False
            else:
                check_dur = True
            valid_notes = is_valid_bar_notes(notes, time_sig, check_match_time_sig=check_dur)
            if not valid_notes:
                d_err = dict(
                    notes=debug_pprint_lst_notes(notes, return_meta=True),
                    time_sig=time_sig,
                    bar_duration=time_sig2bar_dur(time_sig),
                    notes_total_duration=get_notes_duration(notes),
                    notes_with_positive_durations=is_notes_pos_duration(notes),
                    notes_no_overlap=not notes_overlapping(notes),
                    notes_have_gap=notes_have_gap(notes)
                )
                raise ValueError(f'Invalid bar notes at {pl.i(i_bar)}th bar w/ {pl.i(d_err)}')
        return lst_notes

    def notes2quantized_notes(
            self, notes: List[ExtNote], time_sig: TimeSignature, number: int = None
    ) -> List[ExtNote]:
        """
        Approximate notes to the quantization `prec`, by taking the note with majority duration

        .. note:: Notes all have 0 offsets, in the output order

        Expect tuplets to be fully quantized before call - intended for triplets to be untouched after call
        """
        dur_slot = 4 * 2**-self.prec  # In quarter length
        dur_bar = time_sig2bar_dur(time_sig)
        n_slots = dur_bar / dur_slot
        if self.prec == 5 and (time_sig.numerator, time_sig.denominator) == (21, 64) and number == 51:
            # another poor transcription, `LMD::125135`
            # actual #slot is 10.5...
            n_slots = 11
            dur_bar = dur_slot * n_slots
        else:
            assert n_slots.is_integer()
            n_slots = int(n_slots)
        bin_edges = [(i*dur_slot, (i+1)*dur_slot) for i in range(n_slots)]  # All info indexed by note order

        def note2range(note):
            if isinstance(note, tuple):
                strt, end = note[0], note[-1]
                return strt.offset, end.offset + end.duration.quarterLength
            else:
                return note.offset, note.offset + note.duration.quarterLength

        notes_ranges = [note2range(n) for n in notes]
        n_notes = len(notes)

        def get_overlap(low, high, idx_note):
            return min(high, notes_ranges[idx_note][1]) - max(low, notes_ranges[idx_note][0])

        def assign_max_note_idx(low, high, options: Iterable[int]):
            """
            :param low: Lower bound of bin range
            :param high: Upperbound bound of bin range
            :param options: Note candidate indices
            """
            max_opn = max(options, key=lambda i: get_overlap(low, high, i))
            return max_opn if get_overlap(low, high, max_opn) > 0 else None
        idxs_note = [assign_max_note_idx(*edge, range(n_notes)) for edge in bin_edges]
        # Having `None`s should rarely happen, really poor quality of transcription
        # One instance where this warning is raised, see `LMD-cleaned::ABBA - Dancing Queen`
        # Notes goes beyond offset of 4 in 4/4 time sig,
        #   whereas music21 wraps them with mod 4 so the notes become 0-offset
        # those notes are not long enough with the actual 0-offset notes, but are selected as it has higher pitch
        slot_filled_flag = [
            (False if i is None else (get_overlap(*edge, i) > 0)) for edge, i in zip(bin_edges, idxs_note)
        ]
        if not all(slot_filled_flag):
            flag_with_idx = [(i, flag) for i, flag in enumerate(slot_filled_flag)]
            missing_idxs = [
                [i for i, f in pairs]
                for flag, pairs in itertools.groupby(flag_with_idx, key=lambda x: x[1]) if not flag
            ]
            idxs_ends = [(idxs[0], idxs[-1]) for idxs in missing_idxs]
            starts_n_spans = [(i_strt*dur_slot, (i_end-i_strt+1) * dur_slot) for i_strt, i_end in idxs_ends]
            # This will definitely not produce `Fraction`s, which is problematic for json output, see `music_export`
            ts = (time_sig.numerator, time_sig.denominator)
            ranges = [(start, start + span) for start, span in starts_n_spans]
            self.log_warn(
                warn_name=WarnLog.BarNoteGap, bar_num=number, time_sig=ts, precision=self.prec, unfilled_ranges=ranges
            )

        offset = 0
        notes_out = []
        for i, n in compress(idxs_note):
            if i is None:  # In case note missing, fill with Rest
                note_dummy = Rest(duration=m21.duration.Duration(quarterLength=n * dur_slot))
                note_dummy.offset = offset
                notes_out.append(note_dummy)
                offset += note2dur(note_dummy)
            else:
                # for tuplets total duration may still not be quantized yet
                # TODO: seems `for_output` no longer needed, see `note2note_cleaned`
                nt = note2note_cleaned(notes[i], q_len=n*dur_slot, for_output=False)
                if isinstance(nt, tuple):
                    dur_ea = quarter_len2fraction(n*dur_slot) / len(nt)
                    note_tups_out = []
                    for i_, nt_tup in enumerate(nt):
                        nt_tup.offset = offset + dur_ea * i_
                        note_tups_out.append(nt_tup)
                    notes_out.append(tuple(note_tups_out))
                else:
                    nt.offset = offset  # Unroll the offsets
                    notes_out.append(nt)
                offset += note2dur(nt)

        assert not notes_overlapping(notes_out)  # Sanity check
        assert sum(note2dur(n) for n in notes_out) == dur_bar
        return notes_out

    def clean_quantized_tuplets(self, notes: List[ExtNote], num_bar: int) -> List[ExtNote]:
        lst = []
        for nt in notes:
            # If triplet notes turned out quantized, i.e. durations are in powers of 2, turn to normal notes
            if isinstance(nt, tuple) and any(self.pc.note_within_prec(n) for n in nt):
                assert all(self.pc.note_within_prec(n) for n in nt)  # Should be equivalent
                lst.extend(nt)
                offsets_, durs_ = notes2offset_duration(notes)
                self.log_warn(warn_name=WarnLog.TupNoteQuant, bar_num=num_bar, offsets=offsets_, durations=durs_)
            else:
                lst.append(nt)
        return lst

    def __call__(
            self, song: Union[str, Score], exp='mxl', return_meta: bool = False, return_key: bool = False
    ) -> Union[ScoreExt, MusicExtractorOutput]:
        """
        :param song: A music21 Score object, or file path to an MXL file
        :param exp: Export mode, one of ['mxl', 'str', 'id', 'str_join', 'visualize']
            If `mxl`, a music21 Score is returned and written to file
            If `str` or `int`, the corresponding tokens and integer ids are returned as lists
            If `str_join`, the tokens are jointed together
            If `visualize`, a grouped, colorized string is returned, intended for console output
        :param return_meta: If true, metadata about the music is returned, along with the score as a dictionary
            Metadata includes 1) the song title, 2) the song duration in seconds, and 3) warnings found
        :param return_key: If true, possible key signatures of the song is returned
            See `musicnlp.preprocess.key_finder.py`
        """
        t_strt = datetime.datetime.now()
        ca.check_mismatch('Extraction Export Option', exp, ['mxl', 'str', 'id', 'str_join', 'visualize'])
        if self.warn_logger is not None:
            self.warn_logger.end_tracking()

        song_path, song_for_key = None, None
        if isinstance(song, str):
            song_path = song
            song = m21.converter.parse(song)
        song: Score
        if return_key:
            song_for_key = deepcopy(song)  # in case I modified the Score object

        title: str = song.metadata.title or song.metadata.bestTitle
        title = title.removesuffix('.mxl').removesuffix('.musicxml')

        lst_bar_info = list(self.it_bars(song))
        assert len(lst_bar_info) > 0, f'{pl.i("No bars")} found in song'
        assert all(len(bar_info.bars) > 0 for bar_info in lst_bar_info), \
            f'{pl.i("No pitched")} notes found at all times, most likely the song contains {pl.i("drum tracks")} ' \
            f'only - Terminating as extraction output would be empty'
        n_bars_ori = len(lst_bar_info)  # Subject to change, see below

        empty_warns = []
        # Crop out empty bars at both ends to reduce token length
        idx = 0
        while is_empty_bars(lst_bar_info[idx].bars):
            idx += 1
        idx_strt_last_empty = idx-1
        if idx_strt_last_empty != -1:  # 2-tuple, 0-indexed, inclusive on both ends
            empty_warns.append(dict(warn_name=WarnLog.EmptyStrt, bar_range=(0, idx_strt_last_empty)))

        idx = n_bars_ori-1
        while is_empty_bars(lst_bar_info[idx].bars):
            idx -= 1
        idx_end_1st_empty = idx+1
        if idx_end_1st_empty != n_bars_ori:
            empty_warns.append(dict(warn_name=WarnLog.EmptyEnd, bar_range=(idx_end_1st_empty, n_bars_ori-1)))
        lst_bar_info = lst_bar_info[idx_strt_last_empty+1:idx_end_1st_empty]

        lst_bars_, time_sigs, tempos = zip(*(
            (bi.bars, bi.time_sig, bi.tempo) for bi in lst_bar_info
        ))
        # Pick 1st bar arbitrarily
        secs = round(sum(t.durationToSeconds(bars[0].duration) for t, bars in zip(tempos, lst_bars_)))
        mean_tempo = round(np.array([t.number for t in tempos]).mean())  # To the closest integer
        counter_ts = Counter((ts.numerator, ts.denominator) for ts in time_sigs)
        time_sig_mode = max(counter_ts, key=counter_ts.get)
        ts_mode_str = f'{time_sig_mode[0]}/{time_sig_mode[1]}'
        if self.verbose:
            d_log = {'Time Signature': ts_mode_str, 'Tempo': mean_tempo, '#bars': len(lst_bars_), 'Duration': secs}
            self.logger.info(f'Extracting {pl.i(title)} with {pl.i(d_log)}... ')
            if self.warn_logger is not None:
                self.warn_logger.start_tracking(args_func=lambda: dict(id=title, timestamp=now()))
        lst_ts = sorted(set((ts.numerator, ts.denominator) for ts in time_sigs), key=lambda x: (x[1], x[0]))
        lst_tp = sorted(set(round(tp.number) for tp in tempos))
        if len(lst_ts) > 1:
            self.log_warn(warn_name=WarnLog.MultTimeSig, time_sigs=sorted(lst_ts))
        if len(lst_tp) > 1:
            self.log_warn(warn_name=WarnLog.MultTempo, tempos=sorted(lst_tp))
        if not is_common_time_sig(time_sig_mode):
            self.log_warn(warn_name=WarnLog.RareTimeSig, time_sig_expect=COMMON_TIME_SIGS, time_sig_got=time_sig_mode)
        if not is_common_tempo(mean_tempo):
            self.log_warn(warn_name=WarnLog.RareTempo, tempo_expect=COMMON_TEMPOS, tempo_got=mean_tempo)
        for warn_dict in empty_warns:  # Postpone warning message until after logging song info
            self.log_warn(warn_dict)

        th = 0.95
        n_mode, n_bar = counter_ts[time_sig_mode], len(time_sigs)
        if (n_mode / n_bar) < th:  # Arbitrary threshold; Too much invalid time signature
            self.log_warn(
                warn_name=WarnLog.IncTimeSig, time_sig=time_sig_mode, threshold=th, n_bar_total=n_bar, n_bar_mode=n_mode
            )

        d_notes = self.extract_notes(lst_bar_info, time_sigs)
        if exp == 'mxl':
            # unroll tuplets
            d_notes = {k: [list(flatten_notes(notes)) for notes in lst_notes] for k, lst_notes in d_notes.items()}
            scr_out = make_score(
                title=f'{title}\n, extracted', mode=self.mode, time_sig=ts_mode_str, tempo=mean_tempo, d_notes=d_notes,
                check_duration_match=False  # already did it
            )
            dir_nm = sconfig(f'{DSET_DIR}.mxl-eg.dir_nm_extracted')
            fmt = 'mxl'  # sometimes file-writes via `mxl` couldn't be read by MuseScore

            date = now(fmt='short-date')
            mode_str = f'md={self.mode[0]}'
            path = os_join(u.dset_path, dir_nm, f'{date}_{title}_{{{mode_str}}}.{fmt}')
            # disable all `music21` modifications, I should have handled all the edge cases
            scr_out.write(fmt=fmt, fp=path, makeNotation=False)
        else:
            assert exp in ['str', 'id', 'visualize', 'str_join']
            color = exp == 'visualize'
            self.vocab.color = color

            def e2s(elm) -> List[str]:  # Syntactic sugar
                return self.vocab(elm, color=color)

            groups_: List[List[str]] = [[*e2s(time_sig_mode), *e2s(mean_tempo)]]
            if self.mode == 'melody':
                def notes2group(notes: List[SNote]) -> List[str]:
                    return [self.vocab['start_of_bar']] + sum([e2s(n) for n in notes], start=[])
                groups_.extend([notes2group(notes) for notes in d_notes['melody']])
            else:
                def notes2group(notes_melody: List[SNote], notes_bass: List[SNote]) -> List[str]:
                    return (
                        [self.vocab['start_of_bar']] +
                        [self.vocab['start_of_melody']] + sum([e2s(n) for n in notes_melody], start=[]) +
                        [self.vocab['start_of_bass']] + sum([e2s(n) for n in notes_bass], start=[])
                    )
                groups_.extend([notes2group(nm, nb) for nm, nb in zip(d_notes['melody'], d_notes['bass'])])
            groups_.append([self.vocab['end_of_song']])
            if exp == 'visualize':
                n_pad = len(str(len(groups_)))

                def idx2str(i):
                    return pl.s(f'{i:>{n_pad}}:', c='y')
                scr_out = '\n'.join(f'{idx2str(i)} {" ".join(toks)}' for i, toks in enumerate(groups_))
            else:
                toks = sum(groups_, start=[])
                if exp in ['str', 'id']:
                    scr_out = toks if exp == 'str' else self.vocab.encode(toks)
                else:
                    scr_out = ' '.join(toks)
        if self.verbose and self.warn_logger is not None:
            t = fmt_delta(datetime.datetime.now() - t_strt)
            self.logger.info(f'{pl.i(title)} extraction completed in {pl.s(t, c="y")} '
                             f'with warnings {pl.i(self.warn_logger.tracked())}')
        ret = scr_out
        if return_meta:
            warnings = self.warn_logger.tracked(exp='serialize') if self.warn_logger else None
            ret = dict(score=scr_out, title=title, duration=secs, warnings=warnings)
            if song_path:
                ret['song_path'] = song_path
        if return_key:
            keys = KeyFinder(song_for_key)(return_type='dict')
            if isinstance(ret, dict):
                ret['keys'] = keys
            else:
                ret = dict(score=scr_out, keys=keys)
        return MusicExtractorOutput(**ret) if isinstance(ret, dict) else ret


if __name__ == '__main__':
    import re
    import json

    mic.output_width = 512

    import musicnlp.util.music as music_util

    def toy_example():
        logger = WarnLog()
        # fnm = 'Faded'
        # fnm = 'A Thousand Years'
        # fnm = 'Piano Sonata'
        # fnm = 'Ode to Joy'
        # fnm = 'Careless Whisper, 4'
        # fnm = 'Merry Christmas'
        fnm = 'Merry Go Round of Life'
        # fnm = 'Canon piano'
        # fnm = '易燃易爆炸'
        # fnm = 'Shape of You'
        # fnm = '平凡之路'
        # fnm = 'LMD eg'

        # fnm = 'Rolling in the Deep'
        # fnm = "Stayin' Alive"
        # fnm = 'Für Elise'
        # fnm = 'Moonlight'
        # fnm = 'Symphony No.5'
        # fnm = 'Carmen'
        # fnm = 'Señorita'
        # fnm = 'My Heart Will Go On'
        # fnm = 'Ave Maria'
        # fnm = 'Ave Maria (eremita.di.uminho.pt)'
        # fnm = 'Flower Duet'
        # fnm = 'Perfect'
        # fnm = 'Perfect (freemidi)'
        # fnm = 'Perfect (midifilesdownload)'
        # fnm = 'Perfect (cprato)'
        # fnm = 'Hallelujah'
        # fnm = 'Take Me Home Country Roads'
        # fnm = 'Take Me Home Country Roads (freemidi)'
        # fnm = 'Love Yourself'
        # fnm = 'Despacito'
        # fnm = 'Sugar'
        # fnm = 'Beat It'
        # fnm = 'The Marriage of Figaro'
        # fnm = 'Serenade No. 13'
        # fnm = 'KV 448'
        # fnm = 'William Tell'
        # fnm = 'Something Just Like This'
        # fnm = 'Something Just Like This 2'
        # fnm = 'See You Again'

        # fnm = 'Overture from William Tell 2'
        # fnm = 'Autumn Leaves (freemidi)'
        # fnm = 'Autumn Leaves 3'

        # fnm = '走马'
        # fnm = '告白气球'
        # fnm = '演员'
        # fnm = '飘向北方'
        # fnm = '年少有为'
        # fnm = '倒数'
        # fnm = '丑八怪'
        # fnm = '李白'
        # fnm = '挪威的森林'

        # fnm = 'House of the Rising Sun'
        fnm = music_util.get_my_example_songs(fnm, fmt='MXL')
        mic(fnm)
        # mode = 'melody'
        mode = 'full'
        me = MusicExtractor(
            warn_logger=logger, verbose=True, mode=mode, with_pitch_step=True,
            greedy_tuplet_pitch_threshold=16
        )

        def check_mxl_out():
            me(fnm, exp='mxl')
            # mic(logger.to_df())

        def check_str():
            toks = me(fnm, exp='str')
            mic(len(toks), toks[:100])

        def check_visualize():
            s = me(fnm, exp='visualize')
            print(s)

        def check_return_meta_n_key():
            d_out = me(fnm, exp='str_join', return_meta=True, return_key=True)
            mic(d_out)
        # check_mxl_out()
        # check_str()
        check_visualize()
        # check_return_meta_n_key()
    # toy_example()

    def encode_a_few():
        dnm = 'POP909'
        fnms = music_util.get_converted_song_paths(dnm, fmt='mxl')[:]
        # mic(len(fnms), fnms[:5])

        # idx = [idx for idx, fnm in enumerate(fnms) if '恋爱ing' in fnm][0]
        # mic(idx)
        logger = WarnLog()
        me = MusicExtractor(warn_logger=logger, verbose=True, mode='full')
        for i_fl, fnm in enumerate(fnms):
            mic(i_fl, fnm)
            # me(fnm, exp='mxl')
            s = me(fnm, exp='visualize')
            print(s)
    # encode_a_few()

    def profile():
        def func():
            dnm = 'LMD-cleaned-subset'
            fnms = music_util.get_converted_song_paths(dnm, fmt='mxl')[:10]
            me = MusicExtractor(warn_logger=True, verbose=False, greedy_tuplet_pitch_threshold=1)
            for i_fl, fnm in enumerate(fnms):
                mic(i_fl)
                me(fnm, exp='str_join')
        profile_runtime(func)
    # profile()

    def fix_find_song_with_error():
        fnm = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/MNLP-Combined/' \
              'music extraction, 04.09.22_21.13.log'
        with open(fnm, 'r') as f:
            lines = f.readlines()
        mic(len(lines), lines[:5])
        pattern_start = re.compile(r'^.*INFO - Extracting (?P<title>.*) with {.*$')
        pattern_end = re.compile(r'^.*INFO - (?P<title>.*) extraction completed in .*$')
        set_started, set_ended = set(), set()
        for ln in lines:
            m_start, m_end = pattern_start.match(ln), pattern_end.match(ln)
            if m_start:
                set_started.add(m_start.group('title'))
            elif m_end:
                set_ended.add(m_end.group('title'))
        # mic(len(set_started), len(set_ended))
        mic(set_started-set_ended)
    # fix_find_song_with_error()

    def check_edge_case_batched():
        # dnm = 'POP909'
        # dnm = 'MAESTRO'
        dnm = 'LMD'
        if 'LMD' in dnm:
            dir_nm = sconfig(f'datasets.{dnm}.converted.dir_nm')
            dir_nm = f'{dir_nm}, MS'
            # dir_nm = f'{dir_nm}, LP'

            # from _test_broken_files import broken_files
            broken_files = [
                # '035317.mxl',
                # '060134.mxl',
                # '051872.mxl',
                # '067436.mxl',
                # '090283.mxl',
                # '104680.mxl',
                # '107205.mxl',
                # '125135.mxl',
                # '161651.mxl',
                # '173000.mxl',
                # '061641.mxl'
                # '074940.mxl',
                # '084360.mxl',
                # '096500.mxl',
                # '098334.mxl',
                # '107205.mxl',
                # '014187.mxl',
                '068052.mxl',
            ]

            o2f = music_util.Ordinal2Fnm(total=sconfig('datasets.LMD.meta.n_song'), group_size=int(1e4))

            def map_fnm(f: str) -> str:
                _, _dir_nm = o2f(int(stem(f)), return_parts=True)
                return os_join(_dir_nm, f)
            broken_files = [map_fnm(f) for f in broken_files]
        elif dnm == 'MAESTRO':
            broken_files = [
                # "Claude Debussy - L'isle Joyeuse, L. 106.mxl",
                # "Claude Debussy - Les Collines D'anacapri From Book I.mxl",
                # 'Domenico Scarlatti - Sonata In D Major, K. 430.mxl',
                # 'Domenico Scarlatti - Sonata In D Major, K. 96 L. 465.mxl',
                # "Franz Liszt - Annes De Pelerinage Iii: Le Jeux D'eau A La Villa D'este.mxl",
                # 'Franz Liszt - Funerailles.mxl',
                # 'Franz Liszt - Rhapsodie Espagnole, S. 254, v2.mxl',
                # 'Franz Liszt - Rhapsodie Espagnole, S. 254.mxl',
                # 'Franz Liszt - Transcendental Etude No. 10 In F Minor, S. 139:10.mxl',
                # 'Franz Liszt - Tristan And Isolde - Liebestod, S.447.mxl',
                # 'Franz Schubert & Franz Liszt - Song Transcriptions: Aufenthalt, Gretchen Am '
                # 'Spinnrade, Standchen Von Shakespeare, Der Erlkonig, v1.mxl',
                # 'Franz Schubert & Franz Liszt - Song Transcriptions: Aufenthalt, Gretchen Am '
                # 'Spinnrade, Standchen Von Shakespeare, Der Erlkonig, v2.mxl',
                # 'Franz Schubert - Impromptu Op. 142 No. 1, In F Minor, D935.mxl',
                # 'Franz Schubert - Impromptu Op. 142 No. 3 In B-flat Major, D. 935, v1.mxl',
                # 'Franz Schubert - Impromptu Op. 142 No. 3, In B-flat Major, D935, v1.mxl',
                # 'Franz Schubert - Impromptu Op. 90 No. 1, In C Minor, D899, v1.mxl',
                # 'Franz Schubert - Impromptu Op. 90 No. 4 In A-flat Major, v3.mxl'
                'Frédéric Chopin - Variations And Fugue In E-flat Major, Op. 35, "eroica".mxl'
            ]
            dir_nm = 'converted/MAESTRO, MS'
        else:  # POP909
            broken_files = [
                # '许绍洋 - 幸福的瞬间.mxl'
                '五月天 - 天使.mxl',
                '万芳 - 新不了情.mxl',
                '刘德华 - 忘情水.mxl',
                '任贤齐 - 还有我.mxl',
                '孙子涵 - 唐人.mxl',
                '孙子涵 - 回忆那么伤.mxl'
            ]
            dir_nm = 'converted/POP909, MS'
        me = MusicExtractor(warn_logger=True, verbose=True, greedy_tuplet_pitch_threshold=1, mode='full')

        batch = False
        # batch = True

        for broken_fl in broken_files:
            path = os_join(u.dset_path, dir_nm, broken_fl)
            if batch:
                mic(broken_fl)
                try:
                    print(me(path, exp='visualize'))
                except Exception as e:
                    print(pl.s(stem(path), c='y'), e)
                    exit(1)
            else:
                mic(path)
                exp = 'visualize'
                # exp = 'mxl'
                print(me(path, exp=exp))
                exit(1)
    # check_edge_case_batched()

    def fix_find_song_with_0dur():
        """
        Looks like `d_0` is found for some extraction outputs
        """
        dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
                  'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-52-41'
        path = os_join(music_util.get_processed_path(), f'{dnm_lmd}.json')
        mic(path)
        with open(path, 'r') as f:
            dset: Dict = json.load(f)
        songs = dset['music']
        for song in songs:
            txt = song['score']
            if 'd_0' in txt:
                mic(song['title'])
    # fix_find_song_with_0dur()

    def fix_merge_processing_from_lib():
        import glob
        dir_broken_ori = '/Users/stefanhg/Documents/UMich/Research/Music with NLP/datasets/converted/LMD, ' \
                         'broken/020000-030000'
        dir_broken_new = '/Users/stefanhg/Documents/UMich/Research/Music with NLP/datasets/Converted from Lib, ' \
                         '05.22.22/LMD, broken/020000-030000'
        paths_ori, paths_new = glob.iglob(dir_broken_ori + '/*.mid'), glob.iglob(dir_broken_new + '/*.mid')
        set_fls_ori, set_fls_new = set([stem(f) for f in paths_ori]), set([stem(f) for f in paths_new])
        mic(set_fls_ori, set_fls_new)
        mic(set_fls_new - set_fls_ori)
    # fix_merge_processing_from_lib()

    def get_broken_fnms_from_log():
        """
        Get the file names where music extraction failed, by extracting from console output
        """
        import re

        # log_fnm = '05.26.22 @ 17.57, lmd all LP'
        log_fnm = 'MST, 08.02-18.04'
        # path_log = os_join(u.dset_path, 'converted', 'LMD, log', f'{log_fnm}.log')
        path_log = os_join(u.dset_path, 'debug-log', f'{log_fnm}.log')
        with open(path_log, 'r') as f:
            lines = f.readlines()
        pattern = re.compile(r'^.*Failed to extract.*/(?P<fnm>.*).mxl.*$')

        def extract_line(ln: str) -> str:
            m = pattern.match(ln)
            if m:
                return m.group('fnm')
        fnms = [extract_line(ln) for ln in lines]
        fnms = sorted(set([f'{fnm}.mxl' for fnm in fnms if fnm]))
        mic(fnms)
    # get_broken_fnms_from_log()

    def sanity_check_no_consecutive_rests():
        """
        By representation, any consecutive Rest pitches should be merged into a single one
        """
        from tqdm.auto import tqdm
        from musicnlp.vocab import ElmType
        from musicnlp.preprocess import MusicConverter, dataset

        mc = MusicConverter(mode='full')

        pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')
        # dnms = [pop]
        dnms = [pop, mst]
        songs = dataset.load_songs(*dnms)
        rest_meta = MusicVocabulary.step_rest_pitch_meta

        check_implementation = False
        # check_implementation = True
        if check_implementation:
            _fnm_broken = os_join(
                u.dset_path, 'processed', 'intermediate',
                '22-10-22_LMD_{md=f}', '060000-070000', 'Music Export - 068052.json'
            )
            with open(_fnm_broken, 'r') as f:
                _txt_broken = json.load(f)['music']['score']
                # mic(_txt_broken)
        for song in tqdm(songs, unit='song', desc='Sanity check no consecutive rests'):
            txt = song['score']
            if check_implementation:
                txt = _txt_broken
            lst_elms = mc.str2music_elms(txt, pitch_kind='step').elms_by_bar
            for i_bar, elms in enumerate(lst_elms):
                for i, elm in enumerate(elms[:-1]):
                    curr_note_is_rest = elm.type == ElmType.note and elm.meta[0] == rest_meta  # meta for pitch
                    next_note_is_rest = elms[i+1].type == ElmType.note and elms[i+1].meta[0] == rest_meta
                    if curr_note_is_rest and next_note_is_rest:
                        d_log = dict(title=song['title'], bar=i_bar, txt=txt, elm_pair=(elm, elms[i+1]))
                        raise ValueError(f'Consecutive rests found at {pl.i(d_log)}')
                # raise NotImplementedError
    sanity_check_no_consecutive_rests()
