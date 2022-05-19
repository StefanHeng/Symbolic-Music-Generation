"""
Since Sun. Jan. 30th, an updated module for music/melody extraction, with a duration-quantized approach

See `melody_extractor` for the old version.
"""
import math
import datetime
import itertools
from os.path import join as os_join
from copy import deepcopy
from typing import List, Tuple, Dict, Iterable, Iterator, Union, Any
from fractions import Fraction
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


class MusicExtractor:
    """
    Extract melody and potentially chords from MXL music scores => An 1D polyphonic representation
    """
    def __init__(
            self, precision: int = 5, mode: str = 'melody',
            warn_logger: Union[WarnLog, bool] = None,
            greedy_tuplet_pitch_threshold: int = 3**9,
            verbose: Union[bool, str] = True
    ):
        """
        :param precision: Bar duration quantization, see `melody_extractor.MxlMelodyExtractor`
        :param mode: Extraction mode, one of [`melody`, `full`]
            `melody`: Only melody is extracted
            `full`: Melody and Chord as 2 separate channels extracted TODO
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

        self.vocab = MusicVocabulary(precision)

        self.meta = OrderedDict([
            ('mode', mode), ('precision', precision), ('greedy_tuplet_pitch_threshold', greedy_tuplet_pitch_threshold)
        ])

    @staticmethod
    def meta2fnm_meta(d: Dict = None) -> str:
        m, p, t = d['mode'], d['precision'], d['greedy_tuplet_pitch_threshold']
        return log_dict_p({'mode': m, 'prec': p, 'th': t})

    def it_bars(self, scr: Score) -> Iterator[Tuple[Tuple[Measure], TimeSignature, MetronomeMark]]:
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
                assert all(len(t) == 1 for t in tss)
                tss = [next(t) for t in tss]
                assert list_is_same_elms([(ds.numerator, ds.denominator) for ds in tss])
                time_sig = tss[0]

            tempos = [list(b[MetronomeMark]) for b in bars]
            has_tempo = any(tempos)
            if has_tempo:
                tempos = [t for t in tempos if len(t) != 0]
                # When multiple tempos, take the mean
                tempos = [MetronomeMark(number=np.array([t.number for t in ts]).mean()) for ts in tempos]
                bpms = [t.number for t in tempos]
                assert list_is_same_elms(bpms)
                tempo = MetronomeMark(number=bpms[0])
            elif idx == 0:
                self.log_warn(dict(warn_name=WarnLog.MissTempo))
                tempo = MetronomeMark(number=120)  # set as default
            yield [b for ignore, b in zip(ignore, bars) if not ignore], time_sig, tempo

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
        dur_bar = (time_sig.numerator/time_sig.denominator*4)
        n_slots = dur_bar / dur_slot
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

        # if number == 965:
        #     ic(notes)
        #     ic(notes_out)
        #     # for n in flatten_notes(notes):
        #     for n in flatten_notes(notes_out):
        #         qLen = n.duration.quarterLength
        #         ic(n, n.offset, qLen)
        #     # ic([get_overlap(*edge, i) > 0 for edge, i in zip(bin_edges, idxs_note)])
        assert notes_not_overlapping(notes_out)  # Sanity check
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
        # if number == 104:
        #     # ic('in expand_bar', number)
        #     bar.show()
        # if number > 104:
        #     exit(1)
        while elm is not None:
            # full_nm = getattr(elm, 'fullName', None)
            # this is the bottleneck; just care about duration; Explicitly ignore voice
            full_nm = not isinstance(elm, Voice) and getattr(elm.duration, 'fullName', None)
            if full_nm and tuplet_postfix in full_nm:
                tup_str, n_tup = fullname2tuplet_meta(full_nm)

                elms_tup: List[Union[Rest, Note, Chord]] = [elm]
                elm_ = next(it, None)
                while elm_ is not None:
                    # For poor transcription quality, skip over non-note elements in the middle
                    if isinstance(elm_, (m21.clef.Clef, MetronomeMark, m21.bar.Barline)):
                        elm_ = next(it, None)
                    # elif tup_str in elm_.fullName:
                    elif tup_str in elm_.duration.fullName:  # Look for all elements of the same `n_tup`
                        elms_tup.append(elm_)
                        elm_ = next(it, None)  # Peeked 1 ahead
                    else:  # Finished looking for all tuplets
                        break

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
                            if isinstance(note, Chord):
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
                            if not self.dur_within_prec(dur):  # Enforce it by changing the durations
                                warn_nm = WarnLog.InvTupDurSv

                                def round_by_factor(num, fact):
                                    return round(num / fact) * fact
                                # Round to quantization duration; Crop by bar max duration
                                dur = min(
                                    round_by_factor(dur, 2**-self.prec), time_sig.numerator/time_sig.denominator*4
                                )
                                n_tup_last = len(elms_tup[idx_next_strt:])
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
                            lst.append(tuple(elms_tup[idx_next_strt:]))
                            tup_added = True
                            self.log_warn(dict(warn_name=warn_nm, bar_num=number, offsets=offsets, durations=durs))
                    idx += 1
                    e_tup = next(it_tup, None)
                # All triple notes with the same `n_tup` are added
                assert tup_added
                if not is_single_tup:
                    assert sum(len(tup) for tup in lst[idx_tup_strt:]) == len(elms_tup)

                    for tup in lst[idx_tup_strt:]:
                        ln = len(tup)
                        if ln != n_tup:
                            self.log_warn(dict(warn_name=WarnLog.InvTupSz, bar_num=number, n_expect=n_tup, n_got=ln))

                    # Enforce no overlap in each triplet group
                    for idx_tup, tup in enumerate(lst[idx_tup_strt:], start=idx_tup_strt):
                        tup: tuple[Union[Note, Rest]]
                        if not notes_not_overlapping(tup):
                            offsets, durs = notes2offset_duration(tup)
                            self.log_warn(dict(
                                warn_name=WarnLog.TupNoteOvl, bar_num=number, offsets=offsets, durations=durs
                            ))
                            # A really severe case, should rarely happen
                            # TODO: how about just remove this group?
                            # for n in tup:
                            #     name, qLen = n.fullName, n.duration.quarterLength
                            #     ic(name, n.offset, qLen)
                            total_dur: Union[float, Fraction] = sum(n.duration.quarterLength for n in tup)
                            dur_16th = 4 / 16  # duration in quarter length
                            # Trust the duration more than the offset, and unroll backwards to fix the offset
                            # As long as total duration is still multiple of 16th note, make the offset work
                            multiplier: Union[float, Fraction] = total_dur / dur_16th
                            assert multiplier.is_integer() if isinstance(multiplier, float) \
                                else multiplier.denominator == 1
                            note1st = note2note_cleaned(tup[0])
                            offset = note1st.offset + note1st.duration.quarterLength
                            fixed_tup = [note1st]
                            for n in tup[1:]:
                                n = note2note_cleaned(n)
                                n.offset = offset
                                fixed_tup.append(n)
                                offset += n.duration.quarterLength
                            assert notes_not_overlapping(fixed_tup)  # sanity check
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
                                if np.prod(n_opns) > self.greedy_tuplet_pitch_threshold:
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
                elm = elm_
                continue  # Skip `next` for peeked 1 step ahead
            elif isinstance(elm, (Note, Rest)):
                lst.append(elm)
            elif isinstance(elm, Chord):
                if keep_chord:
                    lst.append(elm)
                else:  # TODO: to make things faster, can I keep just the top note?
                    # notes = deepcopy(elm.notes)  # TODO: do I need this?
                    notes = list(elm.notes)
                    for n in notes:
                        n.offset += elm.offset  # Shift offset in the scope of bar
                    lst.extend(notes)
            else:
                if not isinstance(elm, (  # Ensure all relevant types are considered
                    TimeSignature, MetronomeMark, Voice,
                    m21.layout.LayoutBase, m21.clef.Clef, m21.key.KeySignature, m21.bar.Barline
                )):
                    ic(elm)
                    print('unexpected type')
                    exit(1)
            elm = next(it, None)
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

        lst_bar_info: List[tuple[tuple[Measure], TimeSignature, MetronomeMark]] = list(self.it_bars(song))
        assert len(lst_bar_info) > 0, 'No bars found song'
        assert all(len(bar_info[0]) > 0 for bar_info in lst_bar_info), \
            f'No notes found at all times, most likely the song contains {logi("drum tracks")} only - ' \
            f'Terminating as extraction output would be empty'
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
        while is_empty_bars(lst_bar_info[idx][0]):  # grab the all the bars at current number
            idx += 1
        idx_strt_last_empty = idx-1
        if idx_strt_last_empty != -1:  # 2-tuple, 0-indexed, inclusive on both ends
            empty_warns.append(dict(warn_name=WarnLog.EmptyStrt, bar_range=(0, idx_strt_last_empty)))

        idx = n_bars_ori-1
        while is_empty_bars(lst_bar_info[idx][0]):
            idx -= 1
        idx_end_1st_empty = idx+1
        if idx_end_1st_empty != n_bars_ori:
            empty_warns.append(dict(warn_name=WarnLog.EmptyEnd, bar_range=(idx_end_1st_empty, n_bars_ori-1)))
        lst_bar_info = lst_bar_info[idx_strt_last_empty+1:idx_end_1st_empty]

        lst_bars_, time_sigs, tempos = zip(*(
            (bars, time_sig, tempo) for bars, time_sig, tempo in lst_bar_info
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
        set_ts = set(f'{ts.numerator}/{ts.denominator}' for ts in time_sigs)
        set_tp = set(round(tp.number) for tp in tempos)
        if len(set_ts) > 1:
            self.log_warn(dict(warn_name=WarnLog.MultTimeSig, time_sigs=sorted(set_ts)))
        if len(set_tp) > 1:
            self.log_warn(dict(warn_name=WarnLog.MultTempo, tempos=sorted(set_tp)))
        if not is_common_time_sig(time_sig_mode):
            self.log_warn(dict(
                warn_name=WarnLog.UncomTimeSig, time_sig_expect=COMMON_TIME_SIGS, time_sig_got=time_sig_mode
            ))
        if not is_common_tempo(mean_tempo):
            self.log_warn(dict(
                warn_name=WarnLog.UncomTempo, tempo_expect=COMMON_TEMPOS, time_sig_got=mean_tempo
            ))
        for warn_dict in empty_warns:  # Postpone warning message until after logging song info
            self.log_warn(warn_dict)

        th = 0.95
        n_mode, n_bar = counter_ts[time_sig_mode], len(time_sigs)
        if (n_mode / n_bar) < th:  # Arbitrary threshold; Too much invalid time signature
            self.log_warn(dict(
                warn_name=WarnLog.IncTimeSig, time_sig=time_sig_mode, threshold=th, n_bar_total=n_bar, n_bar_mode=n_mode
            ))

        lst_notes: List[List[Union[Note, Chord, tuple[Note]]]] = []  # TODO: melody only
        i_bar_strt = lst_bars_[0][0].number  # Get number of 1st bar
        for i_bar, (bars, time_sig, tempo) in enumerate(lst_bar_info):
            # ic(i_bar)
            number = bars[0].number - i_bar_strt  # Enforce bar number 0-indexing
            assert number == i_bar
            notes = sum((self.expand_bar(b, time_sig, keep_chord=self.mode == 'full', number=number) for b in bars), [])

            groups: Dict[float, List[ExtNote]] = defaultdict(list)  # Group notes by starting location
            for n in notes:
                n_ = n[0] if isinstance(n, tuple) else n
                groups[n_.offset].append(n)
            # if number == 104:
            #     ic(groups)

            def sort_groups():
                for offset, ns in groups.items():  # sort by pitch then by duration, in-place for speed
                    ns.sort(key=lambda nt: (note2pitch(nt), note2dur(nt)))
            sort_groups()

            def get_notes_out() -> List[Union[Note, Chord, tuple[Note]]]:
                # if number == 22:
                #     ic('in new get_notes_out')
                # if not hasattr(get_notes_out, 'recurse_count'):
                #     get_notes_out.recurse_count = 0
                # get_notes_out.recurse_count += 1
                # if get_notes_out.recurse_count % 100 == 0:
                #     ic(get_notes_out.recurse_count)

                ns_out = []
                offset_next = 0
                for offset in sorted(groups.keys()):  # Pass through notes in order
                    notes_ = groups[offset]
                    if len(notes_) == 0:  # As a result of removing triplets
                        del groups[offset]
                        continue
                    nt = notes_[-1]  # Note with the highest pitch
                    nt_ = nt[-1] if isinstance(nt, tuple) else nt
                    nt_end_offset = nt_.offset + nt_.duration.quarterLength
                    # if number == 50:
                    #     ic(ns_out, nt, offset, offset_next)
                    # For difference between floating point and Fraction on real small duration edge cases
                    # See below for another instance
                    if offset_next-offset > eps:
                        # Tuplet notes not normalized at this point, remain faithful to the weighted average pitch
                        if note2pitch(nt) > note2pitch(ns_out[-1]):
                            # Offset would closely line up across tracks, expect this to be less frequent
                            if isinstance(ns_out[-1], tuple):  # triplet being truncated => Remove triplet, start over
                                # The triplet must've been the last note added, and it's joint offset is known
                                del groups[ns_out[-1][0].offset][-1]
                                self.log_warn(dict(warn_name=WarnLog.HighPchOvlTup, bar_num=number))
                                return get_notes_out()
                            else:  # Triplet replaces prior note
                                self.log_warn(dict(warn_name=WarnLog.HighPchOvl, bar_num=number))

                                nt_ = nt[0] if isinstance(nt, tuple) else nt
                                # Resulting duration usually non-0, for offset grouping
                                ns_out[-1].duration = dur_last = Duration(quarterLength=nt_.offset - ns_out[-1].offset)
                                assert dur_last.quarterLength >= 0
                                # If it's 0, it's cos a **truncated** note was appended, as makeup
                                if dur_last.quarterLength == 0:  # TODO: verify
                                    note_2_delete = ns_out.pop()
                                    assert note_2_delete.offset == offset
                                    assert groups[offset][-1] == note_2_delete
                                    del groups[offset][-1]
                                    self.log_warn(dict(warn_name=WarnLog.LowPchMakeupRmv, bar_num=number))
                            ns_out.append(nt)
                            offset_next = nt_end_offset
                        # Later note has smaller pitch, but ends later than the last note
                        # Truncate the note, add it into later group for consideration
                        elif (nt_end_offset-offset_next) > eps:
                            if not isinstance(nt, tuple):
                                # Move the truncated note to later group, restart
                                del groups[offset][-1]
                                nt_ = note2note_cleaned(nt)
                                nt_.offset = offset_next
                                nt_.duration = d = Duration(quarterLength=nt_end_offset-offset_next)
                                assert d.quarterLength > 0
                                offset_next_closest = min(groups.keys(), key=lambda x: abs(x-offset_next))
                                # since Fractions and float that are real close, are not equal (==)
                                if abs(offset_next-offset_next_closest) < eps:
                                    offset_next = offset_next_closest
                                if offset_next in groups:
                                    groups[offset_next].append(nt_)
                                else:
                                    groups[offset_next] = [nt_]
                                sort_groups()
                                self.log_warn(dict(warn_name=WarnLog.LowPchMakeup, bar_num=number))
                                return get_notes_out()
                            # Skip adding tuplets, this potentially leads to gaps in extraction output
                        # Otherwise, skip if later note is lower in pitch and is covered by the prior note duration
                    else:
                        ns_out.append(nt)
                        offset_next = nt_end_offset
                return ns_out

            with RecurseLimit(2**14):
                notes_out = get_notes_out()
            # if number == 965:
            #     ic(notes_out)
            #     assert is_notes_pos_duration(notes_out)
            # For poor transcription quality, postpone `is_valid_bar_notes` *assertion* until after quantization,
            # since empirically observe notes don't sum to bar duration,
            #   e.g. tiny-duration notes shifts all subsequent notes
            #     n: <music21.note.Rest inexpressible>
            #     n.fullName: 'Inexpressible Rest'
            #     n.offset: 2.0
            #     n.duration.quarterLength: Fraction(1, 480)
            dur_bar = time_sig.numerator / time_sig.denominator * 4
            if not math.isclose(sum(n.duration.quarterLength for n in flatten_notes(notes_out)), dur_bar, abs_tol=1e-6):
                offsets, durs = notes2offset_duration(notes_out)
                self.log_warn(dict(  # can be due to removing lower-pitched tuplets
                    warn_name=WarnLog.InvBarDur, bar_num=number, offsets=offsets, durations=durs, time_sig=time_sig
                ))
            if not notes_not_overlapping(notes_out):
                # Convert tuplet to single note by duration, pitch doesn't matter, prep for overlap check
                def tup2note(t: tuple[Note]):
                    note = Note()
                    note.offset = min(note_.offset for note_ in t)
                    q_len_max = max(note_.offset + note_.duration.quarterLength for note_ in t) - note.offset
                    note.duration = Duration(quarterLength=q_len_max)
                    return note
                notes_out_ = [tup2note(n) if isinstance(n, tuple) else n for n in notes_out]  # Temporary, for checking
                assert notes_not_overlapping(notes_out_)  # The source of overlapping should be inside tuplet
                for tup__ in notes_out:
                    if isinstance(tup__, tuple) and not notes_not_overlapping(tup__):
                        offsets, durs = notes2offset_duration(tup__)
                        self.log_warn(dict(
                            warn_name=WarnLog.TupNoteOvl, bar_num=number, offsets=offsets, durations=durs
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
            if not notes_within_prec(notes):
                lst_notes[i_bar] = self.notes2quantized_notes(notes, time_sig, number=i_bar)
                assert notes_within_prec(lst_notes[i_bar])  # Sanity check implementation
                offsets, durs = notes2offset_duration(notes)
                self.log_warn(dict(warn_name=WarnLog.NoteNotQuant, bar_num=i_bar, offsets=offsets, durations=durs))
            elif notes_have_gap(notes):
                lst_notes[i_bar], unfilled_ranges = fill_with_rest(notes)
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
        for notes, time_sig in zip(lst_notes, time_sigs):  # Final check before output
            assert is_valid_bar_notes(notes, time_sig)
        if exp == 'mxl':  # TODO: didn't test
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
        fnm = 'Merry Go Round of Life'
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
        # check_mxl_out()
        # check_str()
        check_visualize()
        # check_return_meta_n_key()
    # toy_example()

    def encode_a_few():
        # dnm = 'POP909'
        dnm = 'LMD-cleaned-subset'
        fnms = music_util.get_cleaned_song_paths(dnm, fmt='mxl')[641:]  # this one too long fnm
        fnms = music_util.get_cleaned_song_paths(dnm, fmt='mxl')[:10]
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
            fnms = music_util.get_cleaned_song_paths(dnm, fmt='mxl')[:10]
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
        dnm = 'MAESTRO'
        dir_nm = sconfig(f'datasets.{dnm}.converted.dir_nm')
        dir_nm = f'{dir_nm}, MS'
        # fnm = '汪峰 - 春天里.mxl'
        # fnm = 'Franz Schubert - Impromptu Op. 142 No. 3 In B-flat Major.mxl'
        fnm = 'Alban Berg - Sonata Op. 1, v2.mxl'
        path = os_join(u.dset_path, dir_nm, fnm)
        me = MusicExtractor(warn_logger=True, verbose=True, greedy_tuplet_pitch_threshold=1)
        # print(me(path, exp='visualize'))
        me(path, exp='mxl')
    check_edge_case()

    def check_edge_case_batched():
        dnm = 'MAESTRO'
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
        broken_files = [
            'Franz Schubert - Sonata In A Major, D. 959 (complete), v2.mxl',
            'Franz Liszt - Après Une Lecture De Dante: Fantasia Quasi Sonata, S.161, No. 7.mxl',
            'Franz Liszt - Transcendental Etude No. 10 In F Minor.mxl',
            'Franz Liszt - Grandes Études De Paganini, No. 3 "la Campanella", S.141:3, v1.mxl',
            'Claude Debussy - Pour Le Piano (complete).mxl',
            'Franz Schubert - Sonata In B-flat Major, D960, v9.mxl',
            'Frédéric Chopin - Etude Op. 10 No. 4 In C-sharp Minor.mxl'
        ]
        me = MusicExtractor(warn_logger=True, verbose=True, greedy_tuplet_pitch_threshold=1)

        for broken_fl in broken_files:
            path = os_join(u.dset_path, dir_nm, broken_fl)
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
