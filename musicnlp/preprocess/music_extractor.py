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
from collections import defaultdict, Counter, OrderedDict

import numpy as np
import music21 as m21

from stefutil import *
from musicnlp.util import *
from musicnlp.util.data_path import BASE_PATH, DSET_DIR
from musicnlp.util.music_lib import *
from musicnlp.vocab import COMMON_TEMPOS, COMMON_TIME_SIGS, is_common_tempo, is_common_time_sig, MusicVocabulary
from musicnlp.preprocess.warning_logger import WarnLog
from musicnlp.preprocess.key_finder import KeyFinder


@dataclass
class BarInfo:
    bars: Union[Tuple[Measure], List[Measure]]
    time_sig: TimeSignature
    tempo: MetronomeMark


class MusicExtractor:
    """
    Extract melody and potentially chords from MXL music scores => An 1D polyphonic representation
    """
    def __init__(
            self, precision: int = 5, mode: str = 'melody',
            warn_logger: Union[WarnLog, bool] = None,
            greedy_tuplet_pitch_threshold: int = 3**9,
            verbose: Union[bool, str] = True,
            epsilon: float = eps
    ):
        """
        :param precision: Bar duration quantization, see `melody_extractor.MxlMelodyExtractor`
        :param mode: Extraction mode, one of [`melody`, `full`]
            `melody`: Only melody is extracted
            `full`: Melody and Bass as 2 separate channels extracted TODO
        :param warn_logger: A logger for storing warnings
            If True, a logger is instantiated
        :param greedy_tuplet_pitch_threshold: If #possible note cartesian product in the tuplet > threshold,
                only the note with highest pitch in chords in tuplets is kept
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
        self.mode = mode

        self.logger = get_logger('Music Extraction')
        assert isinstance(verbose, bool) or verbose == 'single', f'{logi("verbose")} must be bool or {logi("single")}'
        if warn_logger:
            self.warn_logger = warn_logger if isinstance(warn_logger, WarnLog) else WarnLog(verbose=verbose is True)
        else:
            self.warn_logger = None
        self.greedy_tuplet_pitch_threshold = greedy_tuplet_pitch_threshold
        self.verbose = verbose
        self.eps = epsilon

        self.vocab = MusicVocabulary(precision)

        self.meta = OrderedDict([
            ('mode', mode), ('precision', precision), ('greedy_tuplet_pitch_threshold', greedy_tuplet_pitch_threshold)
        ])

    @staticmethod
    def meta2fnm_meta(d: Dict = None) -> str:
        m, p, t = d['mode'], d['precision'], d['greedy_tuplet_pitch_threshold']
        return log_dict_p({'mode': m, 'prec': p, 'th': t})

    def it_bars(self, scr: Score) -> Iterable[BarInfo]:
        """
        Unroll a score by time, with the time signatures of each bar
        """
        parts = list(scr.parts)
        # ic(parts)
        # instrs = list(parts[0][m21.instrument.Instrument])
        # ic(instrs)
        # exit(1)
        ignore = [is_drum_track(p_) for p_ in parts]

        time_sig, tempo = None, None
        for idx, bars in enumerate(zip(*[list(p[Measure]) for p in parts])):  # Bars for all tracks across time
            # Still enumerate the to-be-ignored tracks for getting time signature and tempo
            assert list_is_same_elms([b.number for b in bars]), 'Bar numbers should be the same'

            # Update time signature
            tss = [b[TimeSignature] for b in bars]
            if idx == 0 or any(tss):  # 1st bar must have time signature defined
                assert all(len(t) == 1 for t in tss)
                tss = [next(t) for t in tss]
                assert list_is_same_elms([(ds.numerator, ds.denominator) for ds in tss])
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
                self.log_warn(dict(warn_name=WarnLog.MissTempo))
                tempo = MetronomeMark(number=120)  # set as default
            yield BarInfo(bars=[b for ignore, b in zip(ignore, bars) if not ignore], time_sig=time_sig, tempo=tempo)

    def log_warn(self, log_d: Dict):
        if self.warn_logger is not None:
            self.warn_logger.update(log_d)

    def dur_within_prec(self, dur: Union[float, Fraction]) -> bool:
        return is_int(dur / 4 / (2**-self.prec))

    def notes2quantized_notes(
            self, notes: List[ExtNote], time_sig: TimeSignature, number: int = None
    ) -> List[ExtNote]:
        """
        Approximate notes to the quantization `prec`, by taking the note with majority duration

        .. note:: Notes all have 0 offsets, in the output order

        Expect tuplets to be fully quantized before call - intended for triplets to be untouched after call
        """
        # ic('in quantize', number)
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

            self.log_warn(dict(
                warn_name=WarnLog.BarNoteGap, bar_num=number, time_sig=(time_sig.numerator, time_sig.denominator),
                precision=self.prec, unfilled_ranges=[(start, start+span) for start, span in starts_n_spans]
            ))

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
                nt = note2note_cleaned(notes[i], q_len=n*dur_slot, for_output=True)  # last not processing before output
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

        # if number == 1:
        #     ic(notes)
        #     for n in notes:
        #         strt, end = get_offset(n), get_end_qlen(n)
        #         ic(n, strt, end)
        #     ic(notes_out)
        #     for n in notes_out:
        #         strt, end = get_offset(n), get_end_qlen(n)
        #         ic(n, strt, end)
        #     exit(1)
        #     # ic([get_overlap(*edge, i) > 0 for edge, i in zip(bin_edges, idxs_note)])
        assert not notes_overlapping(notes_out)  # Sanity check
        assert sum(note2dur(n) for n in notes_out) == dur_bar
        return notes_out

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
        # if number == 2:
        #     ic('in expand_bar', number, len(bar))
        #     notes = [e for e in bar if isinstance(e, (Chord, Note, Rest))]
        #     for n in notes:
        #         strt, end = get_offset(n), get_end_qlen(n)
        #         ic(n, n.fullName, strt, end)
        #     bar.show()
        # if number > 17:
        #     exit(1)
        while elm is not None:
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
                    self.log_warn(dict(
                        warn_name=WarnLog.TupNoteOvlIn, bar_num=number, filled_ranges=get_filled_ranges()
                    ))
                if notes_have_gap(elms_tup, enforce_no_overlap=False):
                    self.log_warn(dict(
                        warn_name=WarnLog.TupNoteGap, bar_num=number,
                        time_sig=(time_sig.numerator, time_sig.denominator), filled_ranges=get_filled_ranges()
                    ))

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
                                    self.log_warn(dict(
                                        warn_name=WarnLog.LowTupDur, bar_num=number,
                                        time_sig=(time_sig.numerator, time_sig.denominator),
                                        precision=self.prec,
                                        filled_ranges=get_filled_ranges()
                                    ))
                            if not curr_ignored:
                                lst.append(tuple(elms_tup[idx_next_strt:]))
                                tup_added = True
                            self.log_warn(dict(warn_name=warn_nm, bar_num=number, offsets=offsets, durations=durs))
                    idx += 1
                    e_tup = next(it_tup, None)
                # All triple notes with the same `n_tup` are added
                assert tup_added or tup_ignored
                if not is_single_tup:  # sanity check all notes are processed
                    assert sum(len(tup) for tup in lst[idx_tup_strt:]) + n_ignored == len(elms_tup)

                    for tup in lst[idx_tup_strt:]:
                        ln = len(tup)
                        if ln != n_tup:
                            self.log_warn(dict(warn_name=WarnLog.InvTupSz, bar_num=number, n_expect=n_tup, n_got=ln))

                    # Enforce no overlap in each triplet group
                    for idx_tup, tup in enumerate(lst[idx_tup_strt:], start=idx_tup_strt):
                        tup: tuple[Union[Note, Rest]]
                        if notes_overlapping(tup):
                            offsets, durs = notes2offset_duration(tup)
                            self.log_warn(dict(
                                warn_name=WarnLog.TupNoteOvlOut, bar_num=number, offsets=offsets, durations=durs
                            ))
                            # TODO: how about just remove this group?
                            total_dur: Union[float, Fraction] = sum(n.duration.quarterLength for n in tup)
                            dur_16th = 4 / 16  # duration in quarter length
                            # Trust the duration more than the offset, and unroll backwards to fix the offset
                            # As long as total duration is still multiple of 16th note, make the offset work
                            multiplier: Union[float, Fraction] = total_dur / dur_16th

                            def float_is_int(f: float) -> Union[int, bool]:
                                return f.is_integer() or math.isclose(f, round(f), abs_tol=self.eps)
                            assert float_is_int(multiplier) if isinstance(multiplier, float) \
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
                            self.log_warn(dict(
                                warn_name=WarnLog.RestInTup, bar_num=number, n_rest=n_rest, n_note=len(tup)
                            ))

                    if not keep_chord:
                        tups_new = []
                        has_chord = False
                        for i in range(idx_tup_strt, len(lst)):  # Ensure all tuplet groups contain no Chord
                            tup = lst[i]
                            # Bad transcription quality => Keep all possible tuplet combinations
                            # Try to, but all possible search space is huge as we recurse, see `get_notes_out`
                            # Expect to be the same
                            if any(isinstance(n, Chord) for n in tup):
                                def chord2notes(c):
                                    notes_ = list(c.notes)
                                    for i_ in range(len(notes_)):  # Offsets for notes in chords are 0, restore them
                                        notes_[i_].offset = c.offset
                                    return notes_
                                has_chord = True
                                opns = [chord2notes(n) if isinstance(n, Chord) else (n,) for n in tup]
                                # Adding all possible tuplet notes may be the bottleneck during extraction
                                n_opns = [len(n) for n in opns if n]
                                if math.prod(n_opns) > self.greedy_tuplet_pitch_threshold:
                                    # Too much possible cartesian products for later processing to handle
                                    # as it involves sorting
                                    # Cap at a tuplet of 9 consecutive 3-note Chords, beyond this number,
                                    # just treat the bar as wicked
                                    self.log_warn(dict(
                                        warn_name=WarnLog.ExcecTupNote, bar_num=number, note_choices=n_opns,
                                        threshold=self.greedy_tuplet_pitch_threshold
                                    ))
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
                    # notes = deepcopy(elm.notes)  # TODO: do I need this?
                    notes = list(elm.notes)
                    for n in notes:  # TODO: to make things faster, can I keep just the top note?
                        n.offset += elm.offset  # Shift offset in the scope of bar
                    lst.extend(notes)
            else:
                if not isinstance(elm, (  # Ensure all relevant types are considered
                    TimeSignature, MetronomeMark, Voice,
                    m21.layout.LayoutBase, m21.clef.Clef, m21.key.KeySignature, m21.bar.Barline,
                    m21.expressions.TextExpression
                )):
                    ic(elm)
                    print('unexpected type')
                    exit(1)
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

    def __call__(
            self, song: Union[str, Score], exp='mxl', return_meta: bool = False, return_key: bool = False,
    ) -> Union[ScoreExt, Dict[str, Union[ScoreExt, Any]]]:
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
        exp_opns = ['mxl', 'str', 'id', 'str_join', 'visualize']
        if exp not in exp_opns:
            raise ValueError(f'Unexpected export mode - got {logi(exp)}, expect one of {logi(exp_opns)}')
        if self.warn_logger is not None:
            self.warn_logger.end_tracking()

        song_path, song_for_key = None, None
        if isinstance(song, str):
            song_path = song
            song = m21.converter.parse(song)
        song: Score
        if return_key:  # in case I modified the Score object
            song_for_key = deepcopy(song)

        title = song.metadata.title
        if title.endswith('.mxl'):
            title = title[:-4]

        lst_bar_info = list(self.it_bars(song))
        assert len(lst_bar_info) > 0, f'{logi("No bars")} found in song'
        assert all(len(bar_info.bars) > 0 for bar_info in lst_bar_info), \
            f'{logi("No pitched")} notes found at all times, most likely the song contains {logi("drum tracks")} ' \
            f'only - Terminating as extraction output would be empty'
        n_bars_ori = len(lst_bar_info)  # Subject to change, see below

        # Crop out empty bars at both ends to reduce token length
        def is_empty_bars(bars: tuple[Measure]):
            def bar2elms(b: Measure):
                def stream2elms(stm: Union[Measure, Voice]):
                    return list(chain_its((stm[Note], stm[Rest], stm[Chord])))  # Get all relevant notes
                elms = stream2elms(b)
                if b.hasVoices():
                    elms += sum((stream2elms(v) for v in b.voices), start=[])
                return elms
            return all(all(isinstance(e, Rest) for e in bar2elms(b)) for b in bars)
        empty_warns = []
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
            self.logger.info(f'Extracting {logi(title)} with {log_dict(d_log)}... ')
            if self.warn_logger is not None:
                self.warn_logger.start_tracking(args_func=lambda: dict(id=title, timestamp=now()))
        lst_ts = sorted(set((ts.numerator, ts.denominator) for ts in time_sigs), key=lambda x: (x[1], x[0]))
        lst_tp = sorted(set(round(tp.number) for tp in tempos))
        if len(lst_ts) > 1:
            self.log_warn(dict(warn_name=WarnLog.MultTimeSig, time_sigs=sorted(lst_ts)))
        if len(lst_tp) > 1:
            self.log_warn(dict(warn_name=WarnLog.MultTempo, tempos=sorted(lst_tp)))
        if not is_common_time_sig(time_sig_mode):
            self.log_warn(dict(
                warn_name=WarnLog.UncomTimeSig, time_sig_expect=COMMON_TIME_SIGS, time_sig_got=time_sig_mode
            ))
        if not is_common_tempo(mean_tempo):
            self.log_warn(dict(
                warn_name=WarnLog.UncomTempo, tempo_expect=COMMON_TEMPOS, tempo_got=mean_tempo
            ))
        for warn_dict in empty_warns:  # Postpone warning message until after logging song info
            self.log_warn(warn_dict)

        th = 0.95
        n_mode, n_bar = counter_ts[time_sig_mode], len(time_sigs)
        if (n_mode / n_bar) < th:  # Arbitrary threshold; Too much invalid time signature
            self.log_warn(dict(
                warn_name=WarnLog.IncTimeSig, time_sig=time_sig_mode, threshold=th, n_bar_total=n_bar, n_bar_mode=n_mode
            ))

        lst_notes: List[List[ExtNote]] = []
        i_bar_strt = lst_bars_[0][0].number  # Get number of 1st bar
        for i_bar, bi in enumerate(lst_bar_info):
            bars, time_sig, tempo = bi.bars, bi.time_sig, bi.tempo
            number = bars[0].number - i_bar_strt  # Enforce bar number 0-indexing
            # ic(number)
            assert number == i_bar
            notes = sum((self.expand_bar(b, time_sig, keep_chord=self.mode == 'full', number=number) for b in bars), [])

            groups: Dict[float, List[ExtNote]] = defaultdict(list)  # Group notes by starting location
            for n in notes:
                groups[get_offset(n)].append(n)

            def sort_groups():
                for offset, ns in groups.items():  # sort by pitch then by duration, in-place for speed
                    # ns.sort(key=lambda nt: (note2pitch(nt), note2dur(nt)))
                    # Create shallow copy of list so that no aliasing in list append & removal
                    groups[offset] = sorted(ns, key=lambda nt: (note2pitch(nt), note2dur(nt)))
            sort_groups()

            def _fix_rest_too_long(offset, wrong_end_time):
                _notes_out = []
                for _n in groups[offset]:  # starts at offset 4
                    if isinstance(_n, Rest) and get_end_qlen(_n) == wrong_end_time:  # 4 qlen more than it should
                        continue  # ignore
                    _notes_out.append(_n)
                groups[offset] = _notes_out

            def _fix_edge_case():
                # the original file is broken in that doesn't align with time signature duration
                ts_tup = (time_sig.numerator, time_sig.denominator)
                if ts_tup in [(8, 4), (4, 2), (2, 1)] and \
                        number in [9, 17, 19, 33, 38, 43, 47, 52, 60, 62] and (4.0 in groups or 6.0 in groups):
                    # for [
                    #   `LMD::027213`, `LMD::`050735`, `LMD::054246`, `LMD::069877`, `LMD::108367`,
                    #   `LMD::116976`, `LMD::119887`, `LMD::123389`,
                    # ]
                    for offset in [4.0, 6.0]:
                        _fix_rest_too_long(offset, 12.0)  # 4 more than it should in quarter length
                elif ts_tup == (5, 2) and number in [5, 28] and 6.0 in groups:
                    # for `LMD::109166`
                    _fix_rest_too_long(6.0, 16.0)  # 6 more than it should
                elif ts_tup == (4, 4):
                    if number == 1 and 0.0 in groups:
                        # for `LMD::116496`
                        _notes_out = []
                        for _n in groups[0.0]:
                            wrong_dur = Fraction(33, 8)
                            # assume correct duration of max bar length, 1/8 qlen more than it should
                            if isinstance(_n, tuple) and get_end_qlen(_n) == wrong_dur:
                                _notes_out.append(note2note_cleaned(_n, q_len=4))  # Keep, but normalize
                            else:
                                _notes_out.append(_n)
                        groups[0.0] = _notes_out
                    elif number == 46 and 4.0 in groups:
                        # for `LMD::086800`
                        _notes_out = []
                        for _n in groups[4.0]:  # starts at offset 4
                            e = get_end_qlen(_n)
                            if isinstance(_n, Rest) and \
                                    (math.isclose(e, 4.110416666666667, abs_tol=self.eps) or e == 4.125):
                                continue  # ignore
                            _notes_out.append(_n)
                        groups[4.0] = _notes_out
            _fix_edge_case()

            # if number == 17:
            #     ic('before get notes out', time_sig)
            #     ic(groups)
            #     for k, notes in groups.items():
            #         ic(k)
            #         for n in notes:
            #             strt, end = get_offset(n), get_end_qlen(n)
            #             ic(n, strt, end, n.offset, n.duration)
            #     # exit(1)

            def get_notes_out() -> List[Union[Note, Chord, tuple[Note]]]:
                # if number == 17:
                #     ic('in new get_notes_out')
                #     if not hasattr(get_notes_out, 'recurse_count'):
                #         get_notes_out.recurse_count = 0
                #     get_notes_out.recurse_count += 1
                #     if get_notes_out.recurse_count > 10:
                #         exit(1)
                #     if get_notes_out.recurse_count % 100 == 0:
                #         ic(get_notes_out.recurse_count)

                ns_out = []
                last_end = 0  # Last added note, end time in quarter length
                for offset in sorted(groups.keys()):  # Pass through notes in order
                    notes_ = groups[offset]
                    if len(notes_) == 0:  # As a result of removing triplets
                        del groups[offset]
                        continue
                    nt = notes_[-1]  # Note with the highest pitch
                    nt_end = get_end_qlen(nt)
                    if last_end-offset > self.eps:
                        # Current note starts before the last added note ends
                        # Tuplet notes not normalized at this point, remain faithful to the weighted average pitch
                        note_last = ns_out[-1]
                        pch_last, pch_curr = note2pitch(note_last), note2pitch(nt)
                        if pch_curr > pch_last:  # Truncate last added note
                            if isinstance(note_last, tuple):  # tuplet being truncated => Remove entirely, start over
                                # The triplet must've been the last note added, and it's joint offset is known
                                del groups[get_offset(note_last)][-1]
                                self.log_warn(dict(warn_name=WarnLog.HighPchOvlTup, bar_num=number))
                                return get_notes_out()
                            else:
                                self.log_warn(dict(warn_name=WarnLog.HighPchOvl, bar_num=number))

                                nt_ = nt[0] if isinstance(nt, tuple) else nt
                                # Resulting duration usually non-0, for offset grouping
                                note_last.duration = dur_last = Duration(quarterLength=nt_.offset - note_last.offset)
                                assert dur_last.quarterLength >= 0
                                # If it's 0, it's cos a **truncated** note was appended, as makeup
                                if dur_last.quarterLength == 0:  # TODO: verify
                                    note_2_delete = ns_out.pop()
                                    assert note_2_delete.offset == offset
                                    assert groups[offset][-1] == note_2_delete
                                    del groups[offset][-1]
                                    self.log_warn(dict(warn_name=WarnLog.LowPchMakeupRmv, bar_num=number))
                            ns_out.append(nt)
                            last_end = nt_end
                        # Current note to add has lower pitch, but ends later in time than the last note added
                        # Truncate current note, add back into group based on new start time, recompute
                        elif pch_curr < pch_last and (nt_end-last_end) > self.eps:
                            if not isinstance(nt, tuple):
                                # Move the truncated note to later group, restart
                                del groups[offset][-1]
                                nt_ = note2note_cleaned(nt)
                                # ic(offset, groups[offset], nt_)
                                nt_.offset = last_end
                                nt_.duration = d = Duration(quarterLength=nt_end-last_end)
                                assert d.quarterLength > 0
                                last_end_closest = min(groups.keys(), key=lambda x: abs(x-last_end))
                                # since Fractions and float that are real close, are not equal (==)
                                if abs(last_end-last_end_closest) < self.eps:
                                    last_end = last_end_closest
                                if last_end in groups:
                                    groups[last_end].append(nt_)
                                else:
                                    groups[last_end] = [nt_]
                                sort_groups()
                                self.log_warn(dict(warn_name=WarnLog.LowPchMakeup, bar_num=number))
                                return get_notes_out()
                            # Skip adding tuplets, this potentially leads to gaps in extraction output
                        # Otherwise, skip if later note is lower in pitch and is covered by the prior note duration
                    else:
                        ns_out.append(nt)
                        last_end = nt_end
                return ns_out
            with RecurseLimit(2**14):
                notes_out = get_notes_out()
            # if number == 1:
            #     ic(notes_out)
            #     ic('after notes out')
            #     for n in notes_out[0]:
            #         strt, end = get_offset(n), get_end_qlen(n)
            #         ic(n, strt, end)
            # For poor transcription quality, postpone `is_valid_bar_notes` *assertion* until after quantization,
            # since empirically observe notes don't sum to bar duration,
            #   e.g. tiny-duration notes shifts all subsequent notes
            #     n: <music21.note.Rest inexpressible>
            #     n.fullName: 'Inexpressible Rest'
            #     n.offset: 2.0
            #     n.duration.quarterLength: Fraction(1, 480)
            dur_bar = time_sig.numerator / time_sig.denominator * 4
            note_dur = sum(n.duration.quarterLength for n in flatten_notes(notes_out))
            if not math.isclose(note_dur, dur_bar, abs_tol=self.eps):
                offsets, durs = notes2offset_duration(notes_out)
                self.log_warn(dict(  # can be due to removing lower-pitched tuplets
                    warn_name=WarnLog.InvBarDur, bar_num=number, offsets=offsets, durations=durs, time_sig=time_sig
                ))
            if notes_overlapping(notes_out):
                # Convert tuplet to single note by duration, pitch doesn't matter, prep for overlap check
                def tup2note(t: tuple[Note]):
                    note = Note()
                    note.offset = min(note_.offset for note_ in t)
                    q_len_max = max(note_.offset + note_.duration.quarterLength for note_ in t) - note.offset
                    note.duration = Duration(quarterLength=q_len_max)
                    return note
                notes_out_ = [tup2note(n) if isinstance(n, tuple) else n for n in notes_out]  # Temporary, for checking
                assert not notes_overlapping(notes_out_)  # The source of overlapping should be inside tuplet
                for tup__ in notes_out:
                    if isinstance(tup__, tuple) and notes_overlapping(tup__):
                        offsets, durs = notes2offset_duration(tup__)
                        self.log_warn(dict(
                            warn_name=WarnLog.TupNoteOvlOut, bar_num=number, offsets=offsets, durations=durs
                        ))
            lst_notes.append([note2note_cleaned(n) for n in notes_out])

        # Enforce quantization
        dur_slot = 4 / 2**self.prec  # quarterLength by quantization precision

        def val_within_prec(val: float) -> bool:
            return (val / dur_slot).is_integer()

        def note_within_prec(note):
            return val_within_prec(note2dur(note)) and val_within_prec(get_offset(note))

        def notes_within_prec(notes_):
            return all(note_within_prec(n__) for n__ in notes_)
        for i_bar, (notes, time_sig) in enumerate(zip(lst_notes, time_sigs)):
            dur = time_sig2bar_dur(time_sig)
            if not notes_within_prec(notes):
                lst_notes[i_bar] = self.notes2quantized_notes(notes, time_sig, number=i_bar)
                assert notes_within_prec(lst_notes[i_bar])  # Sanity check implementation
                offsets, durs = notes2offset_duration(notes)
                self.log_warn(dict(warn_name=WarnLog.NoteNotQuant, bar_num=i_bar, offsets=offsets, durations=durs))
            elif notes_have_gap(notes, duration=dur):
                lst_notes[i_bar], unfilled_ranges = fill_with_rest(notes, duration=dur)
                self.log_warn(dict(
                    warn_name=WarnLog.BarNoteGap, bar_num=i_bar, time_sig=(time_sig.numerator, time_sig.denominator),
                    precision=self.prec, unfilled_ranges=unfilled_ranges
                ))
        # Now, triplets fixed to equal duration by `notes2quantized_notes`

        def trip_n_quant2notes(notes_: List[Union[Rest, Note, tuple[Note]]], num_bar: int):
            lst = []
            for nt in notes_:
                # If triplet notes turned out quantized, i.e. durations are in powers of 2, turn to normal notes
                if isinstance(nt, tuple) and any(note_within_prec(n__) for n__ in nt):
                    assert all(note_within_prec(n__) for n__ in nt)  # Should be equivalent
                    lst.extend(nt)
                    offsets_, durs_ = notes2offset_duration(notes)
                    self.log_warn(dict(
                        warn_name=WarnLog.TupNoteQuant, bar_num=num_bar, offsets=offsets_, durations=durs_
                    ))
                else:
                    lst.append(nt)
            return lst
        lst_notes = [trip_n_quant2notes(notes, num_bar=i) for i, notes in enumerate(lst_notes)]

        for notes in lst_notes:
            for n in notes:
                if not isinstance(n, tuple):  # ignore tuplet durations as `consolidate` doesn't consider tuplets
                    # Merges complex durations into one for MXL output
                    n.duration.consolidate()
        for i_bar, (notes, time_sig) in enumerate(zip(lst_notes, time_sigs)):  # Final check before output
            # Edge case, see `notes2quantized_notes`
            check_dur = not (self.prec == 5 and (time_sig.numerator, time_sig.denominator) == (21, 64) and i_bar == 51)

            if not is_valid_bar_notes(notes, time_sig, check_match_time_sig=check_dur):
                ic(i_bar)
                for n in notes:
                    strt, end = get_offset(n), get_end_qlen(n)
                    ic(n, strt, end)

                dur_bar = time_sig2bar_dur(time_sig)
                pos_dur = is_notes_pos_duration(notes)
                no_ovl = not notes_overlapping(notes)
                have_gap = notes_have_gap(notes)
                match_bar_dur = math.isclose(sum(n.duration.quarterLength for n in flatten_notes(notes)), dur_bar,
                                             abs_tol=self.eps)
                ic(pos_dur, no_ovl, (not have_gap), match_bar_dur, time_sig, dur_bar)
                exit(1)
            assert is_valid_bar_notes(notes, time_sig, check_match_time_sig=check_dur)
        if exp == 'mxl':
            scr_out = make_score(
                title=f'{title}, extracted', mode=self.mode, time_sig=ts_mode_str, tempo=mean_tempo,
                lst_note=[list(flatten_notes(notes)) for notes in lst_notes]
            )
            dir_nm = sconfig(f'{DSET_DIR}.mxl-eg.dir_nm_extracted')
            fmt = 'mxl'  # sometimes file-writes via `mxl` couldn't be read by MuseScore
            mode_str = 'melody only' if self.mode == 'melody' else 'full'
            path = os_join(BASE_PATH, DSET_DIR, dir_nm, f'{title}, {mode_str}.{fmt}')
            # disable all `music21` modifications, I should have handled all the edge cases
            scr_out.write(fmt=fmt, fp=path, makeNotation=False)
        else:
            assert exp in ['str', 'id', 'visualize', 'str_join']
            color = exp == 'visualize'
            self.vocab.color = color

            def e2s(elm):  # Syntactic sugar
                return self.vocab(elm, color=color)

            groups_: List[List[str]] = [
                [*e2s(time_sig_mode), *e2s(mean_tempo)],
                *(([self.vocab['start_of_bar']] + sum([e2s(n) for n in notes], start=[])) for notes in lst_notes),
                [self.vocab['end_of_song']]
            ]  # TODO: adding Chords as 2nd part?
            if exp == 'visualize':
                n_pad = len(str(len(groups_)))

                def idx2str(i):
                    return log_s(f'{i:>{n_pad}}:', c='y')
                scr_out = '\n'.join(f'{idx2str(i)} {" ".join(toks)}' for i, toks in enumerate(groups_))
            else:
                toks = sum(groups_, start=[])
                if exp in ['str', 'id']:
                    scr_out = toks if exp == 'str' else self.vocab.encode(toks)
                else:
                    scr_out = ' '.join(toks)
        if self.verbose and self.warn_logger is not None:
            t = fmt_delta(datetime.datetime.now() - t_strt)
            self.logger.info(f'{logi(title)} extraction completed in {log_s(t, c="y")} '
                             f'with warnings {log_dict(self.warn_logger.tracked())}')
        ret = scr_out
        if return_meta:
            ret = dict(score=scr_out, title=title, duration=secs, warnings=self.warn_logger.tracked(exp='serialize'))
            if song_path:
                ret['song_path'] = song_path
        if return_key:
            keys = KeyFinder(song_for_key).find_key(return_type='dict')
            if isinstance(ret, dict):
                ret['keys'] = keys
            else:
                ret = dict(score=scr_out, keys=keys)
        return ret


if __name__ == '__main__':
    import re
    import json

    from icecream import ic

    ic.lineWrapWidth = 512

    import musicnlp.util.music as music_util

    def toy_example():
        logger = WarnLog()
        # fnm = 'Faded'
        # fnm = 'Piano Sonata'
        # fnm = 'Merry Christmas'
        # fnm = 'Merry Go Round of Life'
        fnm = '易燃易爆炸'
        fnm = music_util.get_my_example_songs(fnm, fmt='MXL')
        # fnm = music_util.get_my_example_songs('Shape of You', fmt='MXL')
        # fnm = music_util.get_my_example_songs('平凡之路', fmt='MXL')
        # fnm = music_util.get_my_example_songs('Canon piano')
        ic(fnm)
        me = MusicExtractor(warn_logger=logger, verbose=True)

        def check_mxl_out():
            me(fnm, exp='mxl')
            # ic(logger.to_df())

        def check_str():
            toks = me(fnm, exp='str')
            ic(len(toks), toks[:20])

        def check_visualize():
            s = me(fnm, exp='visualize')
            print(s)

        def check_return_meta_n_key():
            d_out = me(fnm, exp='str_join', return_meta=True, return_key=True)
            ic(d_out)
        check_mxl_out()
        # check_str()
        # check_visualize()
        # check_return_meta_n_key()
    # toy_example()

    def encode_a_few():
        # dnm = 'POP909'
        dnm = 'LMD-cleaned-subset'
        fnms = music_util.get_converted_song_paths(dnm, fmt='mxl')[641:]  # this one too long fnm
        fnms = music_util.get_converted_song_paths(dnm, fmt='mxl')[:10]
        # ic(len(fnms), fnms[:5])

        # idx = [idx for idx, fnm in enumerate(fnms) if '恋爱ing' in fnm][0]
        # ic(idx)
        logger = WarnLog()
        me = MusicExtractor(warn_logger=logger, verbose=True)
        for i_fl, fnm in enumerate(fnms):
            ic(i_fl, fnm)
            me(fnm, exp='mxl')
            # s = mt(fnm, exp='visualize')
            # print(s)
    # encode_a_few()

    def profile():
        def func():
            dnm = 'LMD-cleaned-subset'
            fnms = music_util.get_converted_song_paths(dnm, fmt='mxl')[:10]
            me = MusicExtractor(warn_logger=True, verbose=False, greedy_tuplet_pitch_threshold=1)
            for i_fl, fnm in enumerate(fnms):
                ic(i_fl)
                me(fnm, exp='str_join')
        profile_runtime(func)
    # profile()

    def check_vocabulary():
        vocab = MusicVocabulary()
        ic(vocab.enc, vocab.dec, len(vocab))

        # fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        # mt = MusicTokenizer()
        # toks = mt(fnm, exp='str')
        # ic(vocab.encode(toks[:20]))
    # check_vocabulary()

    def fix_find_song_with_error():
        fnm = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/MNLP-Combined/' \
              'music extraction, 04.09.22_21.13.log'
        with open(fnm, 'r') as f:
            lines = f.readlines()
        ic(len(lines), lines[:5])
        pattern_start = re.compile(r'^.*INFO - Extracting (?P<title>.*) with {.*$')
        pattern_end = re.compile(r'^.*INFO - (?P<title>.*) extraction completed in .*$')
        set_started, set_ended = set(), set()
        for ln in lines:
            m_start, m_end = pattern_start.match(ln), pattern_end.match(ln)
            if m_start:
                set_started.add(m_start.group('title'))
            elif m_end:
                set_ended.add(m_end.group('title'))
        # ic(len(set_started), len(set_ended))
        ic(set_started-set_ended)
    # fix_find_song_with_error()

    def check_edge_case():
        # dnm = 'POP909'
        # dnm = 'MAESTRO'
        dnm = 'LMD'
        dir_nm = sconfig(f'datasets.{dnm}.converted.dir_nm')
        dir_nm = f'{dir_nm}, MS'
        # fnm = '汪峰 - 春天里.mxl'
        # fnm = 'Franz Schubert - Impromptu Op. 142 No. 3 In B-flat Major.mxl'
        # fnm = 'Alban Berg - Sonata Op. 1, v2.mxl'
        # fnm = 'Franz Liszt - Tarantelle Di Bravura, S. 386.mxl'
        # fnm = 'Johann Sebastian Bach - Prelude And Fugue In E Major, Wtc I, Bwv 854.mxl'
        # path = os_join(u.dset_path, dir_nm, fnm)
        # path = '/Users/stefanhg/Desktop/Untitled 186.xml'
        # fnm = '103233.mxl'
        # path = os_join(u.dset_path, dir_nm, '100000-110000', fnm)
        fnm = '128869.mxl'
        path = os_join(u.dset_path, 'converted', 'LMD, check error', fnm)
        me = MusicExtractor(warn_logger=True, verbose=True, greedy_tuplet_pitch_threshold=1)
        # print(me(path, exp='visualize'))
        me(path, exp='mxl')
    check_edge_case()
    # exit(1)

    def check_edge_case_batched():
        # dnm = 'MAESTRO'
        dnm = 'LMD'
        dir_nm = sconfig(f'datasets.{dnm}.converted.dir_nm')
        dir_nm = f'{dir_nm}, MS'
        # broken_files = ['Grandi - Dolcissimo amore', 'John Elton - Burn Down the Mission']
        # broken_files = ['Battiato - Segnali di vita', 'Billy Joel - The River of Dreams']
        # broken_files = ['Pooh - Anni senza fiato', 'Nirvana - Been a Son']
        # broken_files = [
        #     'Daniele - Io vivo come te',
        #     'Drupi - Voglio una donna',
        #     'Flamingos - I Only Have Eyes for You',
        #     'Ozzy Osbourne - Mr. Crowley',
        #     'Ricky Martin - Jaleo (Spanglish)'
        # ]
        # broken_fl = broken_files[0]
        # broken_fl = 'U2 - The Electric Co.'
        # broken_files = [
        #     'Franz Schubert - Sonata In A Major, D. 959 (complete), v2.mxl',
        #     'Franz Liszt - Après Une Lecture De Dante: Fantasia Quasi Sonata, S.161, No. 7.mxl',
        #     'Franz Liszt - Transcendental Etude No. 10 In F Minor.mxl',
        #     'Franz Liszt - Grandes Études De Paganini, No. 3 "la Campanella", S.141:3, v1.mxl',
        #     'Claude Debussy - Pour Le Piano (complete).mxl',
        #     'Franz Schubert - Sonata In B-flat Major, D960, v9.mxl',
        #     'Frédéric Chopin - Etude Op. 10 No. 4 In C-sharp Minor.mxl'
        # ]
        # broken_files = [
        #     # '000065.mxl',
        #     # '000431.mxl',
        #     # '000523.mxl',
        #     # '000338.mxl',
        #     # '000562.mxl',
        #     # '000122.mxl',
        #     # '000211.mxl',
        #     # '000284.mxl',
        #     # '000721.mxl',
        #     # '000186.mxl',
        #     # '000822.mxl',
        #     # '000709.mxl',
        #     # '001139.mxl',
        #     # '001176.mxl',
        #     # '001240.mxl',
        #     # '001489.mxl',
        #     # '001154.mxl',
        #     # '001617.mxl',
        #     # '001128.mxl',
        #     # '001097.mxl',
        #     # '001317.mxl',
        #     # '001909.mxl',
        #     '001764.mxl',
        #     '001549.mxl',
        #     '001803.mxl'
        #     # '000123.mxl',
        #     # '000455.mxl',
        #     # '001144.mxl',
        #     '001282.mxl',
        #     # '001216.mxl'
        #     # '001219.mxl',
        #     # '002380.mxl',
        #     # '002436.mxl',
        #     '002197.mxl'
        #     # '002669.mxl',
        #     # '002810.mxl',
        #     # '002577.mxl',
        #     # '003335.mxl',
        #     # '002888.mxl',
        #     # '003768.mxl',
        #     # '003659.mxl',
        #     # '004368.mxl',
        #     # '004929.mxl',
        #     # '004564.mxl',
        #     # '004875.mxl',
        #     # '004331.mxl',
        #     # '004645.mxl',
        #     '004464.mxl'
        #     # '003348.mxl',
        #     # '005398.mxl',
        #     # '005098.mxl',
        #     # '005340.mxl',
        #     # '005475.mxl',
        #     # '005973.mxl',
        #     # '005747.mxl',
        #     # '006624.mxl',
        #     '006144.mxl'
        #     # '006095.mxl',
        #     # '006637.mxl',
        #     # '006890.mxl',
        #     # '006825.mxl',
        #     # '007025.mxl',
        #     # '005444.mxl',
        #     # '007156.mxl',
        #     # '007860.mxl',
        #     # '007326.mxl',
        #     # '008092.mxl',
        #     # '008605.mxl',
        #     # '008816.mxl',
        #     # '008567.mxl',
        #     '008696.mxl',
        #     # '008816.mxl',
        #     # '009399.mxl',
        #     # '009353.mxl',
        #     # '009483.mxl',
        #     '009858.mxl'
        # ]
        # broken_files = [
        #     # '010853.mxl',
        #     # '010994.mxl',
        #     # '011076.mxl',
        #     # '011299.mxl',
        #     # '011487.mxl',
        #     # '011896.mxl',
        #     # '011804.mxl'
        #     # '011899.mxl',
        #     # '012361.mxl',
        #     # '012544.mxl',
        #     # '012434.mxl',
        #     # '012602.mxl',
        #     # '012493.mxl',
        #     # '012943.mxl',
        #     # '012763.mxl',
        #     # '013013.mxl',
        #     # '013277.mxl',
        #     # '013215.mxl',
        #     # '013629.mxl',
        #     # '013551.mxl',
        #     # '012969.mxl',
        #     # '013989.mxl',
        #     # '014247.mxl',
        #     # '014447.mxl',
        #     # '014391.mxl',
        #     # '014538.mxl',
        #     # '014891.mxl',
        #     # '014964.mxl',
        #     # '015364.mxl',
        #     # '015780.mxl',
        #     # '015976.mxl',
        #     # '015882.mxl',
        #     # '015984.mxl',
        #     # '016304.mxl',
        #     # '016597.mxl',
        #     # '016869.mxl',
        #     # '016932.mxl',
        #     # '017151.mxl',
        #     # '017111.mxl',
        #     # '017228.mxl',
        #     # '017482.mxl',
        #     # '017707.mxl',
        #     # '017948.mxl',
        #     # '018015.mxl',
        #     # '018376.mxl',
        #     # '018622.mxl',
        #     # '017265.mxl',
        #     # '016347.mxl',
        #     # '018901.mxl',
        #     # '019033.mxl',
        #     # '019234.mxl',
        #     # '019297.mxl',
        #     # '019984.mxl',
        #     '019250.mxl',
        #     '019352.mxl',
        #     '018376.mxl',
        #     '018033.mxl',
        # ]
        # broken_files = [
        #     # '020396.mxl',
        #     # '020145.mxl',
        #     # '020358.mxl',
        #     # '020557.mxl',
        #     # '020846.mxl',
        #     # '020683.mxl',
        #     # '021209.mxl',
        #     # '020831.mxl',
        #     # '020257.mxl',
        #     # '021341.mxl',
        #     # '021777.mxl',
        #     # '021912.mxl',
        #     # '022179.mxl',
        #     # '022490.mxl',
        #     # '022986.mxl',
        #     # '022860.mxl',
        #     # '022744.mxl',
        #     # '022576.mxl',
        #     # '021939.mxl',
        #     # '020182.mxl',
        #     # '023977.mxl',
        #     # '023616.mxl',
        #     # '024091.mxl',
        #     # '020846.mxl',
        #     # '021912.mxl',
        #     # '024327.mxl',
        #     # '024653.mxl',
        #     # '024592.mxl',
        #     # '025049.mxl',
        #     # '025591.mxl',
        #     # '025250.mxl',
        #     # '026051.mxl',
        #     # '025967.mxl',
        #     # '024661.mxl',
        #     # '024609.mxl',
        #     # '026132.mxl',
        #     # '026884.mxl',
        #     # '027213.mxl',
        #     # '026751.mxl',
        #     # '027607.mxl',
        #     # '027966.mxl',
        #     # '026884.mxl',
        #     # '027980.mxl',
        #     # '028717.mxl',
        #     # '028285.mxl',
        #     # '027228.mxl',
        #     # '027267.mxl',
        #     # '028371.mxl',
        #     # '029373.mxl',
        #     # '029730.mxl',
        #     # '029873.mxl',
        #     # '029921.mxl',
        #     '029627.mid',  # music21 have trouble parsing this file, takes more than 5 min...
        # ]
        # broken_files = [
        #     # '030110.mxl',
        #     # '030588.mxl',
        #     # '030334.mxl',
        #     # '030369.mxl',
        #     # '030647.mxl',
        #     # '031642.mxl',
        #     # '031337.mxl',
        #     # '031405.mxl',
        #     # '032699.mxl',
        #     # '032582.mxl',
        #     # '032636.mxl',
        #     # '032787.mxl',
        #     # '033045.mxl',
        #     # '032787.mxl',
        #     # '033901.mxl',
        #     # '033514.mxl',
        #     # '034666.mxl',
        #     # '031675.mxl',
        #     # '033922.mxl',
        #     # '033224.mxl',
        #     # '034698.mxl',
        #     # '034822.mxl',
        #     # '034586.mxl',
        #     # '033240.mxl',
        #     # '033717.mxl',
        #     # '034129.mxl',
        #     # '034515.mxl',
        #     # '035125.mxl',
        #     # '034765.mxl',
        #     # '035920.mxl',
        #     # '035282.mxl',
        #     # '035556.mxl',
        #     # '035308.mxl',
        #     # '036148.mxl',
        #     # '036565.mxl',
        #     # '037038.mxl',
        #     # '036936.mxl',
        #     # '037547.mxl',
        #     # '037923.mxl',
        #     # '037376.mxl',
        #     # '037625.mxl',
        #     # '037704.mxl',
        #     # '038208.mxl',
        #     # '038031.mxl',
        #     # '038837.mxl',
        #     # '038670.mxl',
        #     # '038855.mxl',
        #     # '039475.mxl',
        #     # '038696.mxl',
        #     # '039249.mxl',
        #     # '039753.mxl',
        #     # '039703.mxl',
        #     # '039145.mxl',
        #     # '039839.mxl',
        #     # '035611.mxl',
        #     '038670.mxl',
        # ]
        # broken_files = [
        #     # '040459.mxl',
        #     # '040566.mxl',
        #     # '040681.mxl',
        #     # '040965.mxl',
        #     # '041173.mxl',
        #     # '041259.mxl',
        #     # '041349.mxl',
        #     # '041428.mxl',
        #     # '041432.mxl',
        #     # '041512.mxl',
        #     # '041992.mxl',
        #     # '042015.mxl',
        #     '042156.mxl',
        #     '042546.mxl',
        #     '042653.mxl',
        #     '042694.mxl',
        #     '042824.mxl',
        #     '042838.mxl',
        #     '042920.mxl',
        #     '042972.mxl',
        #     '043001.mxl',
        #     '043002.mxl',
        #     '043061.mxl',
        #     '043371.mxl',
        #     '043710.mxl',
        #     '043923.mxl',
        #     '044052.mxl',
        #     '044090.mxl',
        #     '044109.mxl',
        #     '044359.mxl',
        #     '044363.mxl',
        #     '044433.mxl',
        #     '044499.mxl',
        #     '045316.mxl',
        #     '045381.mxl',
        #     '045631.mxl',
        #     '045674.mxl',
        #     '045820.mxl',
        #     '045964.mxl',
        #     '046821.mxl',
        #     '046868.mxl',
        #     '047006.mxl',
        #     '047130.mxl',
        #     '047262.mxl',
        #     '047304.mxl',
        #     '047606.mxl',
        #     '047699.mxl',
        #     '047718.mxl',
        #     '047743.mxl',
        #     '048511.mxl',
        #     '049070.mxl',
        #     '049335.mxl',
        #     '049356.mxl',
        #     '049517.mxl',
        #     '049737.mxl',
        #     '049868.mxl',
        #     '049922.mxl',
        #     '049934.mxl'
        # ]
        # broken_files = [
        #     # '050077.mxl',
        #     # '050305.mxl',
        #     # '050355.mxl',
        #     # '050565.mxl',
        #     # '050621.mxl',
        #     # '050651.mxl',
        #     # '050689.mxl',
        #     # '050703.mxl',
        #     # '050735.mxl',
        #     # '050992.mxl',
        #     # '051354.mxl',
        #     # '051696.mxl',
        #     # '051926.mxl',
        #     # '052001.mxl',
        #     # '052324.mxl',
        #     # '052902.mxl',
        #     # '053020.mxl',
        #     # '053135.mxl',
        #     # '053137.mxl',
        #     # '053165.mxl',
        #     # '053585.mxl',
        #     # '053693.mxl',
        #     # '054246.mxl',
        #     # '054280.mxl',
        #     # '054494.mxl',
        #     # '054923.mxl',
        #     # '055028.mxl',
        #     # '055465.mxl',
        #     # '055770.mxl',
        #     # '055924.mxl',
        #     # '056113.mxl',
        #     # '056122.mxl',
        #     # '056453.mxl',
        #     # '056606.mxl',
        #     # '056617.mxl',
        #     # '056792.mxl',
        #     # '057053.mxl',
        #     # '057853.mxl',
        #     '057584.mxl',
        #     '057374.mxl',
        #     '050980.mxl',
        #     '057285.mxl',
        # ]
        # broken_files = [
        #     # '060780.mxl',
        #     # '061434.mxl',
        #     # '061147.mxl',
        #     # '061621.mxl',
        #     # '061971.mxl',
        #     # '062044.mxl',
        #     # '061235.mxl',
        #     # '062224.mxl',
        #     # '062109.mxl',
        #     # '061274.mxl',
        #     # '062363.mxl',
        #     # '062793.mxl',
        #     # '062523.mxl',
        #     # '063122.mxl',
        #     # '063104.mxl',
        #     # '062372.mxl',
        #     # '063157.mxl',
        #     # '063536.mxl',
        #     # '063749.mxl',
        #     # '063561.mxl',
        #     # '064261.mxl',
        #     # '064722.mxl',
        #     # '064532.mxl',
        #     # '064734.mxl',
        #     # '065726.mxl',
        #     # '066250.mxl',
        #     # '066380.mxl',
        #     # '066585.mxl',
        #     # '066752.mxl',
        #     # '066936.mxl',
        #     # '066876.mxl',
        #     # '067436.mxl',
        #     # '067711.mxl',
        #     # '067526.mxl',
        #     # '067812.mxl',
        #     # '068132.mxl',
        #     # '067765.mxl',
        #     # '068070.mxl',
        #     # '068318.mxl',
        #     # '068042.mxl',
        #     # '068589.mxl',
        #     # '068659.mxl',
        #     # '069017.mxl',
        #     # '067771.mxl',
        #     # '069175.mxl',
        #     # '068166.mxl',
        #     # '068170.mxl',
        #     # '069209.mxl',
        #     # '069877.mxl',
        #     # '069773.mxl',
        #     # '069799.mxl',
        #     # '069929.mxl',
        #     # '068179.mxl',
        #     '069977.mxl',
        # ]
        # broken_files = [
        #     # '070020.mxl',
        #     # '070486.mxl',
        #     # '070431.mxl',
        #     # '070783.mxl',
        #     # '070849.mxl',
        #     # '070906.mxl',
        #     # '071055.mxl',
        #     # '071060.mxl',
        #     # '071573.mxl',
        #     # '070487.mxl',
        #     # '072475.mxl',
        #     # '072279.mxl',
        #     # '072775.mxl',
        #     # '072284.mxl',
        #     # '073249.mxl',
        #     # '074094.mxl',
        #     # '074321.mxl',
        #     # '074436.mxl',
        #     # '073873.mxl',
        #     # '075076.mxl',
        #     # '074856.mxl',
        #     # '074279.mxl',
        #     # '075055.mxl',
        #     # '075831.mxl',
        #     # '075927.mxl',
        #     # '076528.mxl',
        #     # '076763.mxl',
        #     # '076808.mxl',
        #     # '076834.mxl',
        #     # '076901.mxl',
        #     # '076905.mxl',
        #     # '077081.mxl',
        #     # '077162.mxl',
        #     # '077198.mxl',
        #     # '077774.mxl',
        #     # '077797.mxl',
        #     '077511.mxl',
        #     '077999.mxl',
        #     '078118.mxl',
        #     '078146.mxl',
        #     '078395.mxl',
        #     '078439.mxl',
        #     '078570.mxl',
        #     '078642.mxl',
        #     '078668.mxl',
        #     '078695.mxl',
        #     '078982.mxl',
        #     '079223.mxl',
        #     '079373.mxl',
        #     '079555.mxl',
        #     '079696.mxl',
        #     '079824.mxl',
        #     '079877.mxl',
        #     '079941.mxl'
        # ]
        # broken_files = [
        #     # '080403.mxl',
        #     # '080277.mxl',
        #     # '080237.mxl',
        #     # '080121.mxl',
        #     # '081070.mxl',
        #     # '081220.mxl',
        #     # '081142.mxl',
        #     # '081285.mxl',
        #     # '081475.mxl',
        #     # '081645.mxl',
        #     # '081687.mxl',
        #     # '081720.mxl',
        #     # '081731.mxl',
        #     # '081981.mxl',
        #     # '082499.mxl',
        #     # '082875.mxl',
        #     # '082912.mxl',
        #     # '083053.mxl',
        #     # '083091.mxl',
        #     # '083102.mxl',
        #     # '083271.mxl',
        #     # '084002.mxl',
        #     # '084048.mxl',
        #     # '084088.mxl',
        #     # '084171.mxl',
        #     # '084232.mxl',
        #     # '084268.mxl',
        #     # '084402.mxl',
        #     # '084838.mxl',
        #     # '084958.mxl',
        #     # '085070.mxl',
        #     # '085367.mxl',
        #     # '085518.mxl',
        #     # '085744.mxl',
        #     # '085859.mxl',
        #     # '086053.mxl',
        #     # '086212.mxl',
        #     # '086374.mxl',
        #     # '086481.mxl',
        #     # '086592.mxl',
        #     # '086769.mxl',
        #     # '086800.mxl',
        #     '086961.mxl',
        #     '087081.mxl',
        #     '087104.mxl',
        #     '087201.mxl',
        #     '087274.mxl',
        #     '087315.mxl',
        #     '087375.mxl',
        #     '087833.mxl',
        #     '087970.mxl',
        #     '087984.mxl',
        #     '087985.mxl',
        #     '087991.mxl',
        #     '088138.mxl',
        #     '088347.mxl',
        #     '088814.mxl',
        #     '089084.mxl',
        #     '089156.mxl',
        #     '089164.mxl',
        #     '089189.mxl',
        #     '089691.mxl',
        #     '089903.mxl'
        # ]
        # broken_files = [
        #     # '090002.mxl',
        #     # '090017.mxl',
        #     # '090057.mxl',
        #     # '090156.mxl',
        #     # '090212.mxl',
        #     # '090213.mxl',
        #     # '090276.mxl',
        #     # '090892.mxl',
        #     # '090927.mxl',
        #     # '091195.mxl',
        #     # '091285.mxl',
        #     # '091375.mxl',
        #     # '091628.mxl',
        #     # '091640.mxl',
        #     # '091687.mxl',
        #     # '091921.mxl',
        #     # '091936.mxl',
        #     # '092186.mxl',
        #     # '092242.mxl',
        #     # '092288.mxl',
        #     # '092380.mxl',
        #     # '092395.mxl',
        #     # '092405.mxl',
        #     # '092407.mxl',
        #     # '092415.mxl',
        #     # '092597.mxl',
        #     # '093053.mxl',
        #     # '093099.mxl',
        #     # '093106.mxl',
        #     # '093115.mxl',
        #     # '093116.mxl',
        #     # '093227.mxl',
        #     # '093309.mxl',
        #     # '093403.mxl',
        #     # '093455.mxl',
        #     # '093473.mxl',
        #     # '093569.mxl',
        #     # '093760.mxl',
        #     # '094158.mxl',
        #     # '094217.mxl',
        #     # '094253.mxl',
        #     # '094569.mxl',
        #     # '094678.mxl',
        #     # '094953.mxl',
        #     # '095098.mxl',
        #     # '095168.mxl',
        #     # '095381.mxl',
        #     # '095502.mxl',
        #     # '095558.mxl',
        #     # '095601.mxl',
        #     # '095687.mxl',
        #     # '096037.mxl',
        #     # '096044.mxl',
        #     # '096159.mxl',
        #     # '096483.mxl',
        #     # '097172.mxl',
        #     # '097193.mxl',
        #     # '097288.mxl',
        #     # '097335.mxl',
        #     # '097374.mxl',
        #     # '097413.mxl',
        #     # '097452.mxl',
        #     # '097606.mxl',
        #     # '097668.mxl',
        #     # '097768.mxl',
        #     # '097959.mxl',
        #     # '098202.mxl',
        #     # '098631.mxl',
        #     # '098879.mxl',
        #     # '098930.mxl',
        #     # '099166.mxl',
        #     # '099308.mxl',
        #     # '099394.mxl',
        #     # '099481.mxl',
        #     '099641.mxl',
        #     '099675.mxl'
        # ]
        # broken_files = [
        #     # '100067.mxl',
        #     # '100137.mxl',
        #     # '100138.mxl',
        #     # '100201.mxl',
        #     # '100599.mxl',
        #     # '100621.mxl',
        #     # '100950.mxl',
        #     # '101231.mxl',
        #     # '102224.mxl',
        #     # '102291.mxl',
        #     # '102304.mxl',
        #     # '102457.mxl',
        #     # '102670.mxl',
        #     # '102711.mxl',
        #     # '102983.mxl',
        #     # '103155.mxl',
        #     # '103193.mxl',
        #     # '103451.mxl',
        #     # '103456.mxl',
        #     # '103501.mxl',
        #     # '104001.mxl',
        #     # '104162.mxl',
        #     # '104238.mxl',
        #     # '104310.mxl',
        #     # '104374.mxl',
        #     # '104593.mxl',
        #     # '104842.mxl',
        #     # '104915.mxl',
        #     # '104997.mxl',
        #     # '105238.mxl',
        #     # '105584.mxl',
        #     # '105797.mxl',
        #     # '105869.mxl',
        #     # '105993.mxl',
        #     # '106066.mxl',
        #     # '106144.mxl',
        #     # '106178.mxl',
        #     # '106211.mxl',
        #     # '106462.mxl',
        #     # '106607.mxl',
        #     # '106616.mxl',
        #     # '106687.mxl',
        #     # '106823.mxl',
        #     # '106842.mxl',
        #     # '107056.mxl',
        #     # '107103.mxl',
        #     # '107280.mxl',
        #     # '107479.mxl',
        #     # '107647.mxl',
        #     # '107713.mxl',
        #     # '107876.mxl',
        #     # '107946.mxl',
        #     # '107965.mxl',
        #     # '108367.mxl',
        #     # '108513.mxl',
        #     # '108519.mxl',
        #     # '108537.mxl',
        #     # '108841.mxl',
        #     # '109166.mxl',
        #     '109208.mxl',
        #     '109253.mxl',
        #     '109324.mxl',
        #     '109380.mxl',
        #     '109406.mxl',
        #     '109805.mxl'
        # ]
        # broken_files = [
        #     # '110459.mxl',
        #     # '110416.mxl',
        #     # '110302.mxl',
        #     # '111158.mxl',
        #     # '111071.mxl',
        #     # '111375.mxl',
        #     # '111569.mxl',
        #     # '111282.mxl',
        #     # '111350.mxl',
        #     # '111391.mxl',
        #     # '111830.mxl',
        #     # '111735.mxl',
        #     # '112271.mxl',
        #     # '112533.mxl',
        #     # '112685.mxl',
        #     # '112397.mxl',
        #     # '112774.mxl',
        #     # '112780.mxl',
        #     # '112545.mxl',
        #     # '113210.mxl',
        #     # '113381.mxl',
        #     # '113045.mxl',
        #     # '113517.mxl',
        #     # '113620.mxl',
        #     # '113670.mxl',
        #     # '114030.mxl',
        #     # '113694.mxl',
        #     # '114813.mxl',
        #     # '114950.mxl',
        #     # '113387.mxl',
        #     # '115408.mxl',
        #     # '115570.mxl',
        #     # '116052.mxl',
        #     # '116371.mxl',
        #     # '116483.mxl',
        #     # '117028.mxl',
        #     # '116909.mxl',
        #     # '118591.mxl',
        #     # '117429.mxl',
        #     # '118207.mxl',
        #     # '118470.mxl',
        #     # '116600.mxl',
        #     # '118741.mxl',
        #     # '117967.mxl',
        #     # '116802.mxl',
        #     # '116941.mxl',
        #     # '117077.mxl',
        #     # '116496.mxl',
        #     # '118658.mxl',
        #     # '116710.mxl',
        #     # '118664.mxl',
        #     # '116976.mxl',
        #     # '118830.mxl',
        #     # '118307.mxl',
        #     # '119887.mxl',
        #     # '119799.mxl',
        #     # '116496.mxl',
        #     # '116600.mxl',
        #     # '116710.mxl',
        #     # '116802.mxl',
        #     # '116909.mxl',
        #     # '116941.mxl',
        #     # '116976.mxl',
        #     # '117028.mxl',
        #     # '117077.mxl',
        #     # '117429.mxl',
        #     # '117546.mxl',
        #     # '117907.mxl',
        #     # '117967.mxl',
        #     # '118207.mxl',
        #     # '118307.mxl',
        #     # '118429.mxl',
        #     # '118470.mxl',
        #     # '118591.mxl',
        #     # '118658.mxl',
        #     # '118741.mxl',
        #     # '118830.mxl',
        #     # '119537.mxl',
        #     # '119799.mxl',
        #     # '119887.mxl',
        #     '118664.mxl',
        # ]
        # broken_files = [
        #     # '120027.mxl',
        #     # '120177.mxl',
        #     # '120351.mxl',
        #     # '120428.mxl',
        #     # '120602.mxl',
        #     # '121186.mxl',
        #     # '121245.mxl',
        #     # '121280.mxl',
        #     # '121643.mxl',
        #     # '122331.mxl',
        #     # '122359.mxl',
        #     # '122749.mxl',
        #     # '123071.mxl',
        #     # '123171.mxl',
        #     # '123389.mxl',
        #     # '123774.mxl',
        #     # '123867.mxl',
        #     # '123875.mxl',
        #     # '124047.mxl',
        #     # '124190.mxl',
        #     # '124538.mxl',
        #     # '124832.mxl',
        #     # '125135.mxl',
        #     # '125452.mxl',
        #     # '125673.mxl',
        #     # '125805.mxl',
        #     # '125956.mxl',
        #     # '126120.mxl',
        #     # '126193.mxl',
        #     # '126248.mxl',
        #     # '126280.mxl',
        #     # '126873.mxl',
        #     # '126915.mxl',
        #     # '127114.mxl',
        #     # '127423.mxl',
        #     # '127468.mxl',
        #     # '127481.mxl',
        #     # '127490.mxl',
        #     # '127684.mxl',
        #     # '127831.mxl',
        #     # '127864.mxl',
        #     # '128104.mxl',
        #     # '128142.mxl',
        #     # '128203.mxl',
        #     # '128241.mxl',
        #     # '128314.mxl',
        #     # '128330.mxl',
        #     # '128707.mxl',
        #     # '128726.mxl',
        #     # '128869.mxl',  # TODO: check error
        #     '129002.mxl',
        #     '129673.mxl',
        #     '129926.mxl'
        # ]
        broken_files = [
            # '150222.mxl',
            '150366.mxl',
            '150380.mxl',
            '150514.mxl',
            '150530.mxl',
            '150670.mxl',
            '151251.mxl',
            '151464.mxl',
            '151821.mxl',
            '151830.mxl',
            '152197.mxl',
        ]
        # grp_nm = '000000-010000'
        # grp_nm = '010000-020000'
        # grp_nm = '020000-030000'
        # grp_nm = '030000-040000'
        # grp_nm = '040000-050000'
        # grp_nm = '050000-060000'
        # grp_nm = '060000-070000'
        # grp_nm = '070000-080000'
        # grp_nm = '080000-090000'
        # grp_nm = '090000-100000'
        # grp_nm = '100000-110000'
        # grp_nm = '110000-120000'
        # grp_nm = '120000-130000'
        # grp_nm = '130000-140000'
        # grp_nm = '140000-150000'
        grp_nm = '150000-160000'
        # grp_nm = '160000-170000'
        # grp_nm = '170000-17856s1'
        broken_files = [os_join(grp_nm, f) for f in broken_files]
        me = MusicExtractor(warn_logger=True, verbose=True, greedy_tuplet_pitch_threshold=1)

        # batch = False
        batch = True

        for broken_fl in broken_files:
            path = os_join(u.dset_path, dir_nm, broken_fl)
            if batch:
                try:
                    print(me(path, exp='visualize'))
                except Exception as e:
                    print(log_s(stem(path), c='y'), e)
            else:
                ic(path)
                print(me(path, exp='visualize'))
            # me(path, exp='mxl')
    # check_edge_case_batched()

    def fix_find_song_with_0dur():
        """
        Looks like `d_0` is found for some extraction outputs
        """
        dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
                  'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-52-41'
        path = os_join(music_util.get_processed_path(), f'{dnm_lmd}.json')
        ic(path)
        with open(path, 'r') as f:
            dset: Dict = json.load(f)
        songs = dset['music']
        for song in songs:
            txt = song['score']
            if 'd_0' in txt:
                ic(song['title'])
    # fix_find_song_with_0dur()

    def fix_merge_processing_from_lib():
        import glob
        dir_broken_ori = '/Users/stefanhg/Documents/UMich/Research/Music with NLP/datasets/converted/LMD, ' \
                         'broken/020000-030000'
        dir_broken_new = '/Users/stefanhg/Documents/UMich/Research/Music with NLP/datasets/Converted from Lib, ' \
                         '05.22.22/LMD, broken/020000-030000'
        paths_ori, paths_new = glob.iglob(dir_broken_ori + '/*.mid'), glob.iglob(dir_broken_new + '/*.mid')
        set_fls_ori, set_fls_new = set([stem(f) for f in paths_ori]), set([stem(f) for f in paths_new])
        ic(set_fls_ori, set_fls_new)
        ic(set_fls_new - set_fls_ori)
    # fix_merge_processing_from_lib()

    def get_broken_fnms_from_log():
        """
        Get the file names where music extraction failed, by extracting from console output
        """
        import re

        log_fnm = '05.25.22 @ 14.15, lmd grp 16'
        path_log = os_join(u.dset_path, 'converted', 'LMD, log', f'{log_fnm}.log')
        with open(path_log, 'r') as f:
            lines = f.readlines()
        pattern = re.compile(r'^.*Failed to extract.*/(?P<fnm>.*).mxl.*$')

        def extract_line(ln: str) -> str:
            m = pattern.match(ln)
            if m:
                return m.group('fnm')
        fnms = [extract_line(ln) for ln in lines]
        fnms = sorted(set([f'{fnm}.mxl' for fnm in fnms if fnm]))
        ic(fnms)
    # get_broken_fnms_from_log()
