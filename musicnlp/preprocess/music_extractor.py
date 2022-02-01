"""
Since Sun. Jan. 30th, an updated module for music/melody extraction, with a duration-quantized approach

See `melody_extractor` for the old version.
"""

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


class WarnLog:
    """
    Keeps track of warnings in music extraction

    JSON-serializable
    """
    InvTup, HighPch, IncTs = 'Invalid Tuplet', 'Higher Pitch Overlap', 'Inconsistent Time Signatures'
    T_WN = [InvTup, HighPch, IncTs]  # Warning types

    def __init__(self):
        self.warnings: List[Dict] = []

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
        if nm == 'Invalid Tuplet':
            assert all(k in args for k in ['bar_num', 'n_expect', 'n_got'])
        elif nm == 'Higher Pitch Overlap':
            assert 'bar_num' in args
        else:
            assert all(k in args for k in ['time_sig', 'n_bar_total', 'n_bar_mode'])
        self.warnings.append(warn_)

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.warnings)  # TODO: change column names?
        return df


class MusicExtractor:
    """
    Extract melody and potentially chords from MXL music scores => An 1D polyphonic representation
    """
    def __init__(self, scr: Union[str, Score], precision: int = 5, mode: str = 'melody', logger: WarnLog = None):
        """
        :param scr: A music21 Score object, or file path to an MXL file
        :param precision: Bar duration quantization, see `melody_extractor.MxlMelodyExtractor`
        :param mode: Extraction mode, one of [`melody`, `full`]
            `melody`: Only melody is extracted
            `full`: Melody and Chord as 2 separate channels extracted TODO
        :param logger: A logger for processing
        """
        if isinstance(scr, str):
            self.scr = m21.converter.parse(scr)
        else:
            self.scr = scr
        self.scr: Score

        title = self.scr.metadata.title
        if title.endswith('.mxl'):
            title = title[:-4]
        self.title = title

        self.prec = precision
        self.mode = mode

        self.logger = logger

    def expand_bar(
            self, bar: Union[Measure, Voice], keep_chord=False, number=None
    ) -> List[Union[tuple[Note], Rest, Note]]:
        """
        Expand elements in a bar into individual notes, no order is enforced

        :param bar: A music21 measure to expand
        :param keep_chord: If true, `Chord`s are not expanded
        :param number: For passing bar number recursively to Voice

        .. note:: Triplets (potentially any n-plets) are grouped; `Voice`s are expanded
        """
        if not hasattr(MusicExtractor, 'post'):
            MusicExtractor.post = 'plet'  # Postfix for all tuplets, e.g. `Triplet`, `Quintuplet`
        if not hasattr(MusicExtractor, 'post2tup'):
            MusicExtractor.pref2n = dict(  # Tuplet prefix to the expected number of notes
                Tri=3,
                Quintu=5,
                Nonu=9
            )
        post = MusicExtractor.post

        lst = []
        it = iter(bar)
        elm = next(it, None)
        while elm is not None:
            if hasattr(elm, 'fullName') and post in elm.fullName:
                pref = elm.fullName[:elm.fullName.find(post)].split()[-1]
                tup = f'{pref}{post}'
                if pref in MusicExtractor.pref2n:
                    n_tup = MusicExtractor.pref2n[pref]
                else:
                    assert pref == 'Tu'  # A generic case, music21 processing, different from that of MuseScore
                    # e.g. 'C in octave 1 Dotted 32nd Tuplet of 9/8ths (1/6 QL) Note' makes 9 notes in tuplet
                    words = elm.fullName.split()
                    word_n_tup = words[words.index(tup)+2]
                    n_tup = int(word_n_tup[:word_n_tup.find('/')])

                elms_tup: List[Note] = [elm]
                elm_ = next(it, None)
                while elm_ is not None and tup in elm_.fullName:  # Look for all elements of the same `n_tup`
                    elms_tup.append(elm_)
                    elm_ = next(it, None)  # Peeked 1 ahead

                # Consecutive tuplet notes => (potentially multiple) groups
                it_tup = iter(elms_tup)
                e_tup = next(it_tup, None)
                dur: Union[Fraction, float] = 0
                idx, idx_prev, idx_last = 0, 0, len(elms_tup)-1
                n_tup_curr = 0
                trip_added = False
                idx_tup_strt = len(lst)

                def check_wrong_n_tup():
                    ln = len(lst[-1])
                    if ln != n_tup:
                        warn(f'Invalid {tup}: {tup} with invalid number of notes added at bar#{number}'
                             f' - expect {n_tup}, got {ln}')
                        if self.logger is not None:
                            self.logger.update(dict(
                                nm=WarnLog.InvTup, args=dict(bar_num=number, n_expect=n_tup, n_got=ln),
                                id=self.title, timestamp=now()
                            ))
                # MIDI & MuseScore transcription quality, e.g. A triplet may not contain 3 notes
                while e_tup is not None:
                    dur += e_tup.duration.quarterLength
                    n_tup_curr += 1
                    # TODO: generalize beat/tuplet duration checking logic, might involve time signature
                    # Enforce a tuplet must have at least `n_tup` notes
                    # Duration ends as a beat; Heuristic for end of tuplet group
                    if n_tup_curr >= n_tup and dur.denominator == 1:
                        lst.append(tuple(elms_tup[idx_prev:idx+1]))
                        trip_added = True

                        # Prep for next tuplet
                        idx_prev = idx+1
                        n_tup_curr = 0
                        dur = 0

                        if idx == idx_last:  # Postpone warning later, see below
                            check_wrong_n_tup()
                    # Processed til last element, last tuplet group not enough elements
                    if idx == idx_last and n_tup_curr < n_tup:
                        assert trip_added
                        assert dur.denominator == 1
                        lst[-1] = lst[-1] + tuple(elms_tup[idx_prev:])  # Join the prior tuplet group
                        check_wrong_n_tup()
                    idx += 1
                    e_tup = next(it_tup, None)
                # All triple notes with the same `n_tup` are added
                assert sum(len(tup) for tup in lst[idx_tup_strt:]) == len(elms_tup)
                if not keep_chord:
                    tups_new = []
                    has_chord = False
                    for i in range(idx_tup_strt, len(lst)):  # Ensure all tuplet groups contain no Chord
                        tup = lst[i]
                        # Bad transcription quality => Keep all possible tuplet combinations
                        # Expect to be the same
                        if any(isinstance(n, Chord) for n in tup):
                            has_chord = True
                            opns = [tuple(n.notes) if isinstance(n, Chord) else (n,) for n in tup]
                            tups_new.extend(list(itertools.product(*opns)))
                    if has_chord:  # Update prior triplet groups
                        lst = lst[:idx_tup_strt] + tups_new
                if not trip_added:
                    ic('triplet not added')
                    exit(1)
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
            lst.extend(join_its(self.expand_bar(v, number=bar.number) for v in bar.voices))
        return lst

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

    def __call__(self, exp='mxl'):
        scr = deepcopy(self.scr)

        lst_bar_info = list(MusicExtractor.it_bars(scr))  # TODO
        lst_notes: List[List[Union[Note, Chord, tuple[Note]]]] = []  # TODO: melody only
        for n_out, (bars, time_sig, tempo) in enumerate(lst_bar_info):
            number = bars[0].number
            ic(number)
            # if number == 85:
            #     for b in bars:
            #         b.show()
            # n_slots_per_beat, n_slots = time_sig2n_slots(time_sig, self.prec)
            notes = sum((self.expand_bar(b, keep_chord=self.mode == 'full') for b in bars), [])

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

            groups = defaultdict(list)  # Group notes by starting location
            for n in notes:
                n_ = n[0] if isinstance(n, tuple) else n
                groups[n_.offset].append(n)
            # Sort by pitch then by duration
            groups = {
                offset: sorted(ns, key=lambda nt: (note2pitch(nt), note2dur(nt)))
                for offset, ns in groups.items()
            }

            def get_notes_out() -> List[Union[Note, Chord, tuple[Note]]]:
                ns_out = []
                offset_next = 0
                for offset in sorted(groups.keys()):  # Pass through notes in order
                    notes_ = groups[offset]
                    nt = notes_[-1]
                    if number == 85:
                        ic(offset, offset_next)
                    if offset < offset_next:
                        if note2pitch(nt) > note2pitch(ns_out[-1]):
                            # Offset would closely line up across tracks, expect this to be less frequent
                            warn(f'High pitch overlap: later overlapping note with higher pitch observed '
                                 f'at bar#{number} - prior note truncated')
                            if self.logger is not None:
                                self.logger.update(dict(
                                    nm=WarnLog.HighPch, args=dict(bar_num=number),
                                    id=self.title, timestamp=now()
                                ))
                            if isinstance(ns_out[-1], tuple):  # TODO: recomputing notes, if triplet is overlapping
                                ic('triplet being truncated')
                                exit(1)
                            else:  # Triplet replaces pr..           ior note, which is definitely non triplet
                                nt_ = nt[0] if isinstance(nt, tuple) else nt  # Definitely non-0 for offset grouping
                                ns_out[-1].duration = Duration(quarterLength=nt_.offset - ns_out[-1].offset)
                        else:  # Skip if later note is lower in pitch
                            continue
                    ns_out.append(nt)  # Note with the highest pitch
                    nt_ = nt[-1] if isinstance(nt, tuple) else nt
                    offset_next = nt_.offset + nt_.duration.quarterLength
                return ns_out
            notes_out = get_notes_out()
            assert_notes_no_overlap(notes_out)  # Ensure notes cover the entire bar
            n_last = notes_out[-1]
            n_last = n_last[-1] if isinstance(n_last, tuple) else n_last
            assert (n_last.offset + n_last.duration.quarterLength) == (time_sig.numerator / time_sig.denominator * 4)

            def note2note_cleand(note):
                dur = m21.duration.Duration(quarterLength=note.duration.quarterLength)
                if isinstance(note, Note):  # Removes e.g. `tie`s
                    return Note(pitch=m21.pitch.Pitch(midi=note.pitch.midi), duration=dur)
                elif isinstance(note, Rest):
                    return Rest(duration=dur)
                else:
                    assert isinstance(note, Chord)
                    print('clean chord')
                    exit(1)
            lst_notes.append([
                tuple(note2note_cleand(n_) for n_ in n) if isinstance(n, tuple) else note2note_cleand(n)
                for n in notes_out
            ])

        tempo_nums, time_sigs, bars = zip(*[  # Pick 1st bar arbitrarily
            (tempo.number, time_sig, bars[0].duration.quarterLength) for bars, time_sig, tempo in lst_bar_info
        ])
        mean_tempo = round(np.array(tempo_nums).mean())  # To the closest integer
        counter_ts = Counter((ts.numerator, ts.denominator) for ts in time_sigs)
        time_sig_mode = max(counter_ts, key=counter_ts.get)
        th = 0.95
        n_mode, n_bar = counter_ts[time_sig_mode], len(time_sigs)
        if n_mode / n_bar < th:  # Arbitrary threshold; Too much invalid time signature
            warn(f'Inconsistent Time Signatures: ratio of mode time signature below {th}'
                 f' - #mode {n_mode}, #total {n_bar}')
            if self.logger is not None:
                self.logger.update(dict(
                    nm=WarnLog.IncTs, args=dict(time_sig=time_sig_mode, n_bar_total=n_bar, n_bar_mode=n_mode),
                    id=self.title, timestamp = now()
                ))

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
            bar0.insert(TimeSignature(f'{time_sig_mode[0]}/{time_sig_mode[1]}'))

            dir_nm = config(f'{DIR_DSET}.MXL_EG.dir_nm')
            dir_nm = f'{dir_nm}_out'
            scr_out.append(part)
            scr_out.write(fmt='mxl', fp=os.path.join(PATH_BASE, DIR_DSET, dir_nm, f'{title}.mxl'))


if __name__ == '__main__':
    from icecream import ic

    def toy_example():
        logger = WarnLog()
        fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        # fnm = eg_songs('Shape of You', fmt='MXL')
        ic(fnm)
        me = MusicExtractor(fnm, logger=logger)
        me(exp='mxl')
        ic(logger.to_df())
    toy_example()
