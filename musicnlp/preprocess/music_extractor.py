"""
Since Sun. Jan. 30th, an updated module for music/melody extraction, with a duration-quantized approach

See `melody_extractor` for the old version.
"""

import math
from copy import deepcopy
from warnings import warn
from fractions import Fraction
from collections import defaultdict, Counter

from music21.stream import Score, Measure, Voice
from music21.meter import TimeSignature
from music21.tempo import MetronomeMark
from music21.note import Rest, Note
from music21.chord import Chord
from music21.duration import Duration

from musicnlp.util import *


pd.set_option('display.max_columns', None)  # TODO


class WarnLog:
    """
    Keeps track of warnings in music extraction

    JSON-serializable
    """
    InvTupSz, InvTupNt = 'Invalid Tuplet Size', 'Invalid Tuplet Notes'
    InvTupDr, InvTupDrSv = 'Invalid Tuplet Durations', 'Invalid Tuplet Durations, Severe'
    HighPch, IncTs = 'Higher Pitch Overlap', 'Inconsistent Time Signatures'
    HighPchTup = 'Higher Pitch Overlap with Triplet'
    NoteQuant = 'Notes Beyond Quantization'
    T_WN = [  # Warning types
        InvTupSz, InvTupNt,
        InvTupDr, InvTupDrSv,
        HighPch, HighPchTup,
        IncTs,
        NoteQuant
    ]

    def __init__(self):
        self.warnings: List[Dict] = []
        self.idx_track = None
        self.args_func = None

    def update(self, warn_: Dict):
        """
        :param warn_: Dictionary object specifying warning information
            nm: Warning name
            args - Dict: Warning arguments
            id: Warning entry id
            timestamp: Logging timestamp
        """
        assert 'nm' in warn_ and 'args' in warn_
        nm, args = warn_['nm'], warn_['args']

        assert nm in WarnLog.T_WN
        if nm == WarnLog.InvTupSz:
            assert all(k in args for k in ['bar_num', 'n_expect', 'n_got'])
        elif nm in [
            WarnLog.InvTupNt, WarnLog.InvTupDr, WarnLog.InvTupDrSv,
            WarnLog.NoteQuant
        ]:
            assert all(k in args for k in ['offsets', 'durations'])

        elif nm in [WarnLog.HighPch, WarnLog.HighPchTup]:
            assert 'bar_num' in args
        else:  # IncTs
            assert nm == WarnLog.IncTs
            assert all(k in args for k in ['time_sig', 'n_bar_total', 'n_bar_mode'])
        if self.args_func is not None:
            warn_ = self.args_func() | warn_
        self.warnings.append(warn_)

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.warnings)  # TODO: change column names?
        return df

    def start_tracking(self, args_func: Callable[[], Dict] = None):
        """
        Start tracking warnings added after the call

        :args_func: A function that returns a dictionary of metadata, merged to each `update` call
        """
        self.idx_track = len(self.warnings)
        self.args_func = args_func

    def show_track(self) -> str:
        """
        Statistics of warnings since tracking started
        """
        counts = Counter(w['nm'] for w in self.warnings[self.idx_track:])
        return ', '.join((f'{logi(k)}: {logi(v)}' for k, v in counts.items()))


class MusicTokenizer:
    """
    Extract melody and potentially chords from MXL music scores => An 1D polyphonic representation
    """
    SPEC_TOKS = dict(
        sep='_',  # Separation
        rest='r',
        prefix_pitch='p',
        prefix_duration='d',
        start_of_tuplet='<tup>',
        end_of_tuplet='</tup>',
        start_of_bar='<bar>',
        end_of_song='</s>',
        prefix_time_sig='TimeSig',
        prefix_tempo='Tempo'
    )

    def __init__(self, precision: int = 5, mode: str = 'melody', logger: WarnLog = None, verbose=False):
        """
        :param precision: Bar duration quantization, see `melody_extractor.MxlMelodyExtractor`
        :param mode: Extraction mode, one of [`melody`, `full`]
            `melody`: Only melody is extracted
            `full`: Melody and Chord as 2 separate channels extracted TODO
        :param logger: A logger for processing
        :param verbose: If true, process is logged, including statistics of score and warnings
        """
        self.title = None  # Current score title

        self.prec = precision
        self.mode = mode

        self.logger = logger
        self.verbose = verbose

    @staticmethod
    def it_bars(scr: Score) -> Iterator[tuple[Measure, TimeSignature, MetronomeMark]]:
        """
        Unroll a score by time, with the time signatures of each bar
        """
        # Remove drum tracks
        def is_drum(part):
            """
            :return: True if `part` contains *only* `Unpitched`
            """
            return list(part[m21.note.Unpitched]) and not list(part[m21.note.Note])
        instrs_drum = [
            m21.instrument.BassDrum,
            m21.instrument.BongoDrums,
            m21.instrument.CongaDrum,
            m21.instrument.SnareDrum,
            m21.instrument.SteelDrum,
            m21.instrument.TenorDrum,
        ]
        parts = [p_ for p_ in scr.parts if not (any(p_[drum] for drum in instrs_drum) or is_drum(p_))]

        time_sig, tempo = None, None
        for idx, bars in enumerate(zip(*[list(p[Measure]) for p in parts])):  # Bars for all tracks across time
            assert_list_same_elms([b.number for b in bars])  # Bar numbers should be the same

            # Update time signature
            tss = [b[TimeSignature] for b in bars]
            if idx == 0 or any(tss):  # 1st bar must have time signature defined
                assert all(len(t) == 1 for t in tss)
                tss = [next(t) for t in tss]
                assert_list_same_elms([(ds.numerator, ds.denominator) for ds in tss])
                time_sig = tss[0]

            tempos = [b[MetronomeMark] for b in bars]
            if idx == 0 or any(tempos):
                tempos = [t for t in tempos if len(t) != 0]
                # When multiple tempos, take the mean
                tempos = [MetronomeMark(number=np.array([t.number for t in ts]).mean()) for ts in tempos]
                bpms = [t.number for t in tempos]
                assert_list_same_elms(bpms)

                tempo = MetronomeMark(number=bpms[0])
            yield bars, time_sig, tempo

    def my_log_warn(self, warn_msg: str, log_dict: Dict):
        warn(warn_msg)  #
        if self.logger is not None:
            self.logger.update(log_dict)

    def dur_within_prec(self, dur: Union[float, Fraction]) -> bool:
        # ic(dur / 4 / (2**-self.prec))
        return is_int(dur / 4 / (2**-self.prec))

    def notes2quantized_notes(
            self, notes: Iterable[Union[Note, Rest]], time_sig: TimeSignature
    ) -> List[Union[Note, Rest]]:
        """
        Approximate notes to the quantization `prec`, by taking the note with majority duration

        .. note:: Notes all have 0 offsets, in the output order
        """

        dur_ea = 4 * 2**-self.prec  # In quarter length
        # ic(dur_ea)
        n_slots = 1 / dur_ea * (time_sig.numerator/time_sig.denominator*4)
        assert n_slots.is_integer()
        n_slots = int(n_slots)
        bin_edges = [(i*dur_ea, (i+1)*dur_ea) for i in range(n_slots)]  # All info indexed by note order

        # for n in notes:
        #     ic('ori', n.duration.quarterLength)
        notes = unroll_notes(notes)
        # ic('in quantize')
        # for n in notes:
        #     ic(n, n.offset, n.duration.quarterLength)
        notes_ranges = [(n.offset, n.offset+n.duration.quarterLength) for n in notes]
        ic(notes_ranges, bin_edges)
        n_notes = len(notes)

        # def get_notes_in_bin(low, high):
        #     # return [n for i, n in enumerate(notes) if notes_ranges[i][0] <= high or low <= notes_ranges[i][1]]
        #     return [i for i in range(n_notes) if (notes_ranges[i][0] <= high or low <= notes_ranges[i][1])]
        # bins = [get_notes_in_bin(*edges) for edges in bin_edges]
        # ic(bins, len(bins))

        def assign_max_note_idx(low, high, options: Iterable[int]):
            """
            :param low: Lower bound of bin range
            :param high: Upperbound bound of bin range
            :param options: Note candidate indices
            """
            def get_overlap(i):
                return min(high, notes_ranges[i][1]) - max(low, notes_ranges[i][0])
            return max(options, key=get_overlap)
        idxs_note = [assign_max_note_idx(*edge, range(n_notes)) for edge in bin_edges]
        # idxs_n_slots = compress(idxs_note)
        # ic(idxs_note, idxs_n_slots)
        notes_out = [note2note_cleand(notes[i], q_len=n*dur_ea) for i, n in compress(idxs_note)]

        def sanity_check():
            for n in notes:
                ic('ori', n.duration.quarterLength)
            for n in notes_out:
                ic('quant', n.duration.quarterLength)
            bar_ori = Measure()
            bar_ori.append(notes)
            bar_quant = Measure()
            bar_quant.append(notes_out)
            bar_ori.show()
            bar_quant.show()
        # exit(1)

        return notes_out

    def expand_bar(
            self, bar: Union[Measure, Voice], time_sig: TimeSignature, keep_chord=False, number=None
    ) -> List[Union[tuple[Note], Rest, Note]]:
        """
        Expand elements in a bar into individual notes, no order is enforced

        :param bar: A music21 measure to expand
        :param time_sig: Time signature of the bar
        :param keep_chord: If true, `Chord`s are not expanded
        :param number: For passing bar number recursively to Voice

        .. note:: Triplets (potentially any n-plets) are grouped; `Voice`s are expanded
        """
        if number is None:  # For debugging
            number = bar.number
        # ic('in expand', number)

        if not hasattr(MusicTokenizer, 'post'):
            MusicTokenizer.post = 'plet'  # Postfix for all tuplets, e.g. `Triplet`, `Quintuplet`
        if not hasattr(MusicTokenizer, 'post2tup'):
            MusicTokenizer.pref2n = dict(  # Tuplet prefix to the expected number of notes
                Tri=3,
                Quintu=5,
                Nonu=9
            )
        post = MusicTokenizer.post

        lst = []
        it = iter(bar)
        elm = next(it, None)
        while elm is not None:
            if hasattr(elm, 'fullName') and post in elm.fullName:
                pref = elm.fullName[:elm.fullName.find(post)].split()[-1]
                tup: str = f'{pref}{post}'
                if pref in MusicTokenizer.pref2n:
                    n_tup = MusicTokenizer.pref2n[pref]
                else:
                    assert pref == 'Tu'  # A generic case, music21 processing, different from that of MuseScore
                    # e.g. 'C in octave 1 Dotted 32nd Tuplet of 9/8ths (1/6 QL) Note' makes 9 notes in tuplet
                    words = elm.fullName.split()
                    word_n_tup = words[words.index(tup)+2]
                    n_tup = int(word_n_tup[:word_n_tup.find('/')])

                elms_tup: List[Note] = [elm]
                elm_ = next(it, None)
                while elm_ is not None:
                    # For poor transcription quality, skip over non-note elements in the middle
                    if isinstance(elm_, m21.clef.TrebleClef):
                        # ic('not here')
                        elm_ = next(it, None)
                    elif tup in elm_.fullName:  # Look for all elements of the same `n_tup`
                        elms_tup.append(elm_)
                        elm_ = next(it, None)  # Peeked 1 ahead
                        # ic('here', elm_)
                    else:  # Finished looking for all tuplets
                        break
                if number == 83:
                    ic(elms_tup)
                    for n in elms_tup:
                        ic(n, n.offset, n.duration.quarterLength)
                        ic(n.fullName)
                    ic(sum(n.duration.quarterLength for n in elms_tup))

                # Consecutive tuplet notes => (potentially multiple) groups
                it_tup = iter(elms_tup)
                e_tup = next(it_tup, None)
                dur: Union[Fraction, float] = 0
                idx, idx_prev, idx_last = 0, 0, len(elms_tup)-1
                n_tup_curr = 0
                tup_added = False
                idx_tup_strt = len(lst)
                is_single_tup = False  # Edge case, see below

                def check_wrong_n_tup():
                    ln = len(lst[-1])
                    if ln != n_tup:
                        msg = f'Invalid {tup} sizes: {tup} with invalid number of notes added at bar#{number}' \
                              f' - expect {n_tup}, got {ln}'
                        self.my_log_warn(msg, dict(
                                nm=WarnLog.InvTupSz, args=dict(bar_num=number, n_expect=n_tup, n_got=ln)
                        ))
                # MIDI & MuseScore transcription quality, e.g. A triplet may not contain 3 notes
                while e_tup is not None:
                    dur += e_tup.duration.quarterLength
                    n_tup_curr += 1
                    # TODO: generalize beat/tuplet duration checking logic, might involve time signature
                    # Enforce a tuplet must have at least `n_tup` notes
                    # Duration for a tuplet must be multiples of 8th note; Heuristic for end of tuplet group
                    if n_tup_curr >= n_tup and is_8th(dur):
                        lst.append(tuple(elms_tup[idx_prev:idx+1]))
                        tup_added = True

                        # Prep for next tuplet
                        idx_prev = idx+1
                        n_tup_curr = 0
                        dur = 0

                        if idx == idx_last:  # Postpone warning later, see below
                            check_wrong_n_tup()
                    # Processed til last element, last tuplet group not enough elements
                    if idx == idx_last and n_tup_curr < n_tup:
                        if len(elms_tup) == 1:  # Poor processing, edge case
                            note = elms_tup[0]  # As if single note with weird duration
                            # TODO: if selected later, quantize
                            lst.append(note)
                            tup_added, is_single_tup = True, True
                            e_tup = None
                            continue  # Break out: Done processing this tuplet group
                        if number == 91:
                            ic(elms_tup, dur)
                            for n in elms_tup:
                                ic(n, n.fullName, n.offset, n.duration.quarterLength)
                            ic('bar')
                            for n in bar:
                                ic(n, n.fullName, n.offset, n.duration.quarterLength)
                        if tup_added:
                            lst[-1] = lst[-1] + tuple(elms_tup[idx_prev:])  # Join the prior tuplet group
                        else:  # Number of tuplet notes not enough, create new tuplet anyway
                            tup_added = True
                            lst.append(tuple(elms_tup[idx_prev:]))
                        check_wrong_n_tup()
                    if idx == idx_last and not tup_added:
                        # Must be that duration of all triplet elements don't sum up to an 8th note in quarterLength
                        assert not is_8th(dur)  # Add such tuplets anyway for a valid quantized length
                        if not self.dur_within_prec(dur):  # Enforce it by changing the durations
                            # dur = sum(n.duration.quarterLength for n in elms_tup)
                            def round_by_factor(num, fact):
                                return round(num / fact) * fact
                            # Max duration allowed
                            dur = min(round_by_factor(dur, 2**-self.prec), time_sig.numerator/time_sig.denominator*4)
                            n_tup = len(elms_tup)
                            dur_ea = dur / n_tup
                            strt = elms_tup[0].offset
                            for i in range(n_tup):
                                elms_tup[i].offset = strt
                                elms_tup[i].duration = m21.duration.Duration(quarterLength=dur_ea)
                                strt += dur_ea
                        lst.append(tuple(elms_tup))
                        tup_added = True
                        offsets, durs = zip(*[(n.offset, n.duration.quarterLength) for n in elms_tup])
                        msg = f'Invalid {tup} durations: {tup} durations don\'t sum up to 8th notes ' \
                              f'- notes quantized and cropped to bar length if necessary'
                        self.my_log_warn(msg, dict(
                            nm=WarnLog.InvTupDr, args=dict(bar_num=number, offsets=offsets, durations=durs)
                        ))
                    idx += 1
                    e_tup = next(it_tup, None)
                # All triple notes with the same `n_tup` are added
                assert tup_added
                if not is_single_tup:
                    if number == 83:
                        ic(tup_added, idx_tup_strt)
                        ic(lst[idx_tup_strt:], lst)
                        for tup in lst[idx_tup_strt:]:
                            ic(tup)
                    assert sum(len(tup) for tup in lst[idx_tup_strt:]) == len(elms_tup)

                    for tup in lst[idx_tup_strt:]:  # Enforce on overlap in each triplet group
                        if not is_notes_no_overlap(tup):
                            tup: tuple[Note, Rest, Chord]
                            offsets, durs = zip(*[(n.offset, n.duration.quarterLength) for n in tup])
                            for n in tup:
                                ic(n, n.fullName, n.offset, n.duration.quarterLength)
                            msg = f'Invalid {tup} notes: {tup} with overlapping notes added at bar#{number}, ' \
                                  f'with offsets: {offsets}, durations: {durs} - notes are cut off'
                            self.my_log_warn(msg, dict(
                                nm=WarnLog.InvTupNt, args=dict(bar_num=number, offsets=offsets, durations=durs)
                            ))
                            it_n = iter(tup)
                            nt = next(it_n)
                            end = nt.offset + nt.duration.quarterLength
                            tup_new = [nt]
                            # bar.show()
                            exit(1)
                            # while nt is not None:

                    if not keep_chord:
                        tups_new = []
                        has_chord = False
                        for i in range(idx_tup_strt, len(lst)):  # Ensure all tuplet groups contain no Chord
                            tup = lst[i]
                            # Bad transcription quality => Keep all possible tuplet combinations
                            # Expect to be the same
                            if any(isinstance(n, Chord) for n in tup):
                                def chord2notes(c):
                                    notes_ = list(c.notes)
                                    for i_ in range(len(notes_)):  # Offsets for notes in chords are 0, restore them
                                        notes_[i_].offset = c.offset
                                    return notes_
                                has_chord = True
                                opns = [chord2notes(n) if isinstance(n, Chord) else (n,) for n in tup]
                                tups_new.extend(list(itertools.product(*opns)))
                        if has_chord:  # Replace prior triplet groups
                            lst = lst[:idx_tup_strt] + tups_new
                elm = elm_
                continue  # Skip `next` for peeked 1 step ahead
            elif isinstance(elm, (Note, Rest)):
                lst.append(elm)
            elif isinstance(elm, Chord):
                if keep_chord:
                    lst.append(elm)
                else:
                    notes = deepcopy(elm.notes)
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
            lst.extend(join_its(self.expand_bar(v, time_sig, number=bar.number) for v in bar.voices))
        return lst

    def __call__(self, scr: Union[str, Score], exp='mxl') -> Union[Score, str]:
        """
        :param scr: A music21 Score object, or file path to an MXL file
        :param exp: Export mode, one of [`mxl`, `str_id`, `str_id_color`]
        """
        if isinstance(scr, str):
            scr = m21.converter.parse(scr)
        scr: Score

        title = scr.metadata.title
        if title.endswith('.mxl'):
            title = title[:-4]
        self.title = title

        lst_bar_info = list(MusicTokenizer.it_bars(scr))  # TODO
        tempos, time_sigs, lst_bars_ = zip(*[
            (tempo, time_sig, bars) for bars, time_sig, tempo in lst_bar_info
        ])
        # Pick 1st bar arbitrarily
        secs = round(sum(t.durationToSeconds(bars[0].duration) for t, bars in zip(tempos, lst_bars_)))
        mean_tempo = round(np.array([t.number for t in tempos]).mean())  # To the closest integer
        counter_ts = Counter((ts.numerator, ts.denominator) for ts in time_sigs)
        time_sig_mode = max(counter_ts, key=counter_ts.get)
        ts_mode_str = f'{time_sig_mode[0]}/{time_sig_mode[1]}'
        if self.verbose:
            log(f'Tokenizing music {logi(self.title)}'
                f' - Time signature {logi(ts_mode_str)}, avg Tempo {logi(mean_tempo)}, '
                f'{logi(len(lst_bars_))} bars with Duration {logi(sec2mmss(secs))}...')
            if self.logger is not None:
                self.logger.start_tracking(args_func=lambda: dict(id=self.title, timestamp=now()))

        def note2pitch(note):
            if isinstance(note, tuple):  # Triplet, return average pitch
                # Duration for each note not necessarily same duration, for transcription quality
                fs, durs = zip(*[(note2pitch(n__), n__.duration.quarterLength) for n__ in note])
                return np.average(fs, weights=durs)
            elif isinstance(note, Note):
                return note.pitch.frequency
            else:
                assert isinstance(note, Rest)
                return 0  # `Rest` given pitch frequency of 0

        def note2dur(note):
            if isinstance(note, tuple):
                return sum(note2dur(nt) for nt in note)
            else:
                return note.duration.quarterLength
        lst_notes: List[List[Union[Note, Chord, tuple[Note]]]] = []  # TODO: melody only
        for n_out, (bars, time_sig, tempo) in enumerate(lst_bar_info):
            number = bars[0].number
            ic(number)
            # if number == 85:
            #     for b in bars:
            #         b.show()
            # n_slots_per_beat, n_slots = time_sig2n_slots(time_sig, self.prec)
            notes = sum((self.expand_bar(b, time_sig, keep_chord=self.mode == 'full') for b in bars), [])

            groups = defaultdict(list)  # Group notes by starting location
            for n in notes:
                n_ = n[0] if isinstance(n, tuple) else n
                groups[n_.offset].append(n)
            # Sort by pitch then by duration
            groups = {
                offset: sorted(ns, key=lambda nt: (note2pitch(nt), note2dur(nt)))
                for offset, ns in groups.items()
            }
            # if number == 55:
            #     ic(groups)

            def get_notes_out() -> List[Union[Note, Chord, tuple[Note]]]:
                ns_out = []
                offset_next = 0
                for offset in sorted(groups.keys()):  # Pass through notes in order
                    notes_ = groups[offset]
                    # if number == 55:
                    #     ic(offset, notes_, groups[3.5])
                    nt = notes_[-1]
                    if offset < offset_next:
                        if note2pitch(nt) > note2pitch(ns_out[-1]):
                            # Offset would closely line up across tracks, expect this to be less frequent
                            if isinstance(ns_out[-1], tuple):  # triplet being truncated => Remove triplet, start over
                                # ic(ns_out[-1][0].offset, note2dur(ns_out[-1]), ns_out)
                                # The triplet must've been the last note added, and it's joint offset is known
                                del groups[ns_out[-1][0].offset][-1]
                                msg = f'High Pitch Overlap on Triplet: Higher pitch observed at bar#{number}' \
                                      f' - triplet truncated'
                                self.my_log_warn(msg, dict(nm=WarnLog.HighPchTup, args=dict(bar_num=number)))
                                return get_notes_out()
                            else:  # Triplet replaces prior note, which is definitely non triplet
                                msg = f'High Pitch Overlap: Later overlapping note with higher pitch observed ' \
                                      f'at bar#{number} - prior note truncated'
                                self.my_log_warn(msg, dict(nm=WarnLog.HighPch, args=dict(bar_num=number)))

                                nt_ = nt[0] if isinstance(nt, tuple) else nt  # Definitely non-0 for offset grouping
                                ns_out[-1].duration = Duration(quarterLength=nt_.offset - ns_out[-1].offset)
                        else:  # Skip if later note is lower in pitch
                            continue
                    ns_out.append(nt)  # Note with the highest pitch
                    nt_ = nt[-1] if isinstance(nt, tuple) else nt
                    offset_next = nt_.offset + nt_.duration.quarterLength
                return ns_out
            notes_out = get_notes_out()
            assert is_notes_no_overlap(notes_out)  # Ensure notes cover the entire bar
            n_last = notes_out[-1]
            n_last = n_last[-1] if isinstance(n_last, tuple) else n_last
            assert (n_last.offset + n_last.duration.quarterLength) == (time_sig.numerator / time_sig.denominator * 4)
            lst_notes.append([
                tuple(note2note_cleand(n_) for n_ in n) if isinstance(n, tuple) else note2note_cleand(n)
                for n in notes_out
            ])
        # Enforce quantization
        dur_slot = 4 / 2**self.prec  # quarterLength by quantization precision

        def note_within_prec(note):
            return (note2dur(note) / dur_slot).is_integer()
        for i_bar, (notes, time_sig) in enumerate(zip(lst_notes, time_sigs)):  # TODO: what if smaller than `prec`?
            # ic(i_bar)
            # for n in notes:
            #     if i_bar == 84:
            #         # for b in lst_bar_info[i_bar][0]:
            #         #     # b.show()
            #         #     ic('bar')
            #         #     for n_ in b:
            #         #         ic(n_, n_.fullName, n_.offset, n_.duration.quarterLength)
            #         ic(note2dur(n), dur_slot, note2dur(n) / dur_slot)
            #         ic('')
            #
            #         for nt in notes:
            #             ic(nt, nt.fullName, nt.offset, nt.duration.quarterLength)
            #         ic(sum(n.duration.quarterLength for n in notes))
            #     assert note_within_prec(n)
            if not all(note_within_prec(n) for n in notes):
                lst_notes[i_bar] = self.notes2quantized_notes(notes, time_sig)
                offsets, durs = zip(*[(n.offset, n.duration.quarterLength) for n in unroll_notes(notes)])
                msg = f'Notes Beyond Quantization: Note durations smaller than quantized slot at bar#{i_bar} ' \
                      f'- Note durations approximated'
                self.my_log_warn(msg, dict(
                    nm=WarnLog.NoteQuant, args=dict(bar_num=i_bar, offsets=offsets, durations=durs)
                ))

        th = 0.95
        n_mode, n_bar = counter_ts[time_sig_mode], len(time_sigs)
        if n_mode / n_bar < th:  # Arbitrary threshold; Too much invalid time signature
            msg = f'Inconsistent Time Signatures: ratio of mode time signature below {th}' \
                  f' - #mode {n_mode}, #total {n_bar}'
            self.my_log_warn(msg, dict(
                nm=WarnLog.IncTs, args=dict(time_sig=time_sig_mode, n_bar_total=n_bar, n_bar_mode=n_mode)
            ))
        if self.verbose and self.logger is not None:
            log(f'Encoding {logi(self.title)} completed - Observed warnings {{{self.logger.show_track()}}}')

        if exp == 'mxl':
            scr_out = Score()
            scr_out.insert(m21.metadata.Metadata())
            post = 'Melody only' if self.mode == 'melody' else 'Melody & Chord'
            title = f'{self.title}, {post}'
            scr_out.metadata.title = title
            scr_out.metadata.composer = PKG_NM

            part_nm = 'Melody, Ch#1'  # TODO: a 2nd chord part
            part = m21.stream.Part(partName=part_nm)
            part.partName = part_nm
            instr = m21.instrument.Piano()
            part.append(instr)

            lst_bars = []
            for i, notes in enumerate(lst_notes):
                bar = Measure(number=i)  # Original bar number may not start from 0
                bar.append(list(flatten_notes(notes)))
                lst_bars.append(bar)
            part.append(lst_bars)

            bar0 = part.measure(0)  # Insert metadata into 1st bar
            bar0.insert(MetronomeMark(number=mean_tempo))
            bar0.insert(TimeSignature(ts_mode_str))

            dir_nm = config(f'{DIR_DSET}.MXL_EG.dir_nm')
            dir_nm = f'{dir_nm}_out'
            scr_out.append(part)
            scr_out.write(fmt='mxl', fp=os.path.join(PATH_BASE, DIR_DSET, dir_nm, f'{title}.mxl'))
            return scr_out
        elif exp in ['str_id', 'str_id_color']:
            color = exp == 'str_id_color'
            sep, pref_dur, pref_pitch, rest, bot, eot, bar_sep, end, pref_time_sig, pref_tempo = (
                MusicTokenizer.SPEC_TOKS[k] for k in [
                    'sep', 'prefix_duration', 'prefix_pitch', 'rest',
                    'start_of_tuplet', 'end_of_tuplet', 'start_of_bar', 'end_of_song', 'prefix_time_sig', 'prefix_tempo'
                ]
            )
            pref_dur = f'{pref_dur}{sep}'
            pref_pitch = f'{pref_pitch}{sep}'

            rest = f'{pref_pitch}{rest}'

            time_sig = f'{pref_time_sig}{sep}{ts_mode_str}'
            tempo = f'{pref_tempo}{sep}{mean_tempo}'
            # bar_sep = f' {bar_sep} '
            if color:
                bot, eot = logs(bot, c='m'), logs(eot, c='m')
                bar_sep = logs(bar_sep, c='m')
                end = logs(end, c='m')
                time_sig = logs(time_sig, c='m')
                tempo = logs(tempo, c='m')
                rest = logs(rest, c='b')

            def elm2str(elm: Union[Note, Rest, tuple[Note]]) -> List[str]:  # TODO: Support chords?
                """
                Each relevant token into a string representation
                """
                def note2dur_str(e: Union[Rest, Note, tuple[Note]]) -> str:
                    dur = Fraction(note2dur(e))
                    if dur.denominator == 1:
                        s = f'{pref_dur}{dur.numerator}'
                    else:
                        s = f'{pref_dur}{dur.numerator}/{dur.denominator}'
                    return logs(s, c='g') if color else s

                def pch2step(p: m21.pitch.Pitch) -> int:
                    """
                    Naive mapping to the physical, mod-12 pitch frequency, in [1-12]
                    """
                    s = p.midi % 12
                    return 12 if s == 0 else s+1

                def note2pch_str(note: Note) -> str:
                    pitch = note.pitch
                    # s = f'{pref_pitch}{pitch.name}/{pitch.octave}'
                    s = f'{pref_pitch}{pch2step(pitch)}/{pitch.octave}'
                    return logs(s, c='b') if color else s

                if isinstance(elm, Rest):
                    return [rest, note2dur_str(elm)]
                elif isinstance(elm, Note):
                    return [note2pch_str(elm), note2dur_str(elm)]
                elif isinstance(elm, tuple):
                    # Sum duration for all tuplets
                    return [bot] + [note2pch_str(e) for e in elm] + [note2dur_str(elm), eot]
                else:  # TODO: chords
                    ic('other element type', elm)
                    exit(1)
            # TODO: adding Chords as 2nd part?
            return ' '.join([time_sig, tempo, bar_sep, f' {bar_sep} '.join([
                    (' '.join(join_its(elm2str(n) for n in notes))) for notes in lst_notes
                ]), end
            ])


if __name__ == '__main__':
    from icecream import ic

    def toy_example():
        logger = WarnLog()
        fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        # fnm = eg_songs('Shape of You', fmt='MXL')
        # fnm = eg_songs('平凡之路', fmt='MXL')
        ic(fnm)
        mt = MusicTokenizer(logger=logger, verbose=True)

        def check_mxl_out():
            mt(fnm, exp='mxl')
            ic(logger.to_df())

        def check_str():
            s = mt(fnm, exp='str_id')
            toks = s.split()
            ic(len(toks), toks[:20])

        def check_str_color():
            s = mt(fnm, exp='str_id_color')
            print(s)

        check_mxl_out()
        # check_str()
        # check_str_color()
    # toy_example()

    def encode_a_few():
        dnm = 'POP909'
        fnms = fl_nms(dnm, k='song_fmt_exp')
        # ic(fnms[:20])

        logger = WarnLog()
        mt = MusicTokenizer(logger=logger, verbose=True)
        for fnm in fnms[6:20]:
            # log(f'{dnm} - {os.path.basename(fnm)}')
            mt(fnm)
    encode_a_few()

