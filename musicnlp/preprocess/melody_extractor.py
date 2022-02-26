import math
from copy import deepcopy
from warnings import warn
from fractions import Fraction

from musicnlp.util import *

import numpy as np
from mido import MidiFile
from pretty_midi import PrettyMIDI
import music21 as m21


def bars2lst_bar_n_ts(bars) -> list[tuple[m21.stream.Measure, m21.meter.TimeSignature]]:
    """
    :return: List of tuple of corresponding time signature for each bar
    """
    bars = iter(bars)
    bar0 = next(bars)
    ts = next(bar0[m21.meter.TimeSignature])
    lst_bar_n_ts = [(bar0, ts)]

    for bar in bars:
        ts_ = bar[m21.meter.TimeSignature]
        if ts_:
            ts = next(iter(ts_))
        lst_bar_n_ts.append((bar, ts))
    return lst_bar_n_ts


def invalid_triplets(scr: m21.stream.Score):
    def _invalid(stream: Union[m21.stream.Measure, m21.stream.Voice]):
        # ic()
        it = it_m21_elm(stream, types=(m21.note.Note, m21.note.Rest, m21.chord.Chord, m21.stream.Voice))
        elm = next(it, None)
        while elm:
            # if isinstance(elm, m21.stream.Voice):
            #     ic(elm, elm.offset, elm.duration)
            # else:
            #     # ic(stream.number)
            #     ic(elm, elm.fullName, elm.offset, elm.duration)

            if isinstance(elm, m21.stream.Voice):
                # ic(stream.number)
                if _invalid(elm):
                    return True
            elif 'Triplet' in elm.fullName:
                elm2, elm3 = next(it, None), next(it, None)
                if elm2 is None or elm3 is None:
                    return True
                if 'Triplet' not in elm2.fullName or 'Triplet' not in elm3.fullName:
                    return True
                if not (elm.duration.quarterLength == elm2.duration.quarterLength == elm3.duration.quarterLength):
                    return True
            elm = next(it, None)
        return False
    return any(any(_invalid(b) for b in p[m21.stream.Measure]) for p in scr.parts)


def multiple_clef(scr: m21.stream.Score):
    def _invalid(bar: m21.stream.Measure):
        return sum(isinstance(c, m21.clef.Clef) for c in bar) > 1
    return any(any(_invalid(b) for b in p[m21.stream.Measure]) for p in scr.parts)


def has_quintuplet(scr):
    # Expect rarely seen: Fails the time-slot encoding
    return any(any('Quintuplet' in n.fullName for n in p[m21.note.Note]) for p in scr.parts)


def time_sig2ratio(time_sig: m21.meter.TimeSignature):
    """
    Intended for MXL processing
    """
    # Unit length of time signature over unit length per `quarterLength`
    return time_sig.denominator / 4


class MidiMelodyExtractor:
    """
    Given MIDI file, export single-track melody representations, as matrix or MIDI file

    Only MIDI files with no tempo change are considered

    Velocity of all notes assumed the same

    For now, enforce that at each time step, there will be only one note played

    The pitch of each note follows the MIDI standard, as integer in [0, 127]

    - attributes::
        - frac_per_beat:
            Fraction of the beat, specified by the time signature
            e.g. 4/4 makes quarter note, frac_per_beat <- 4
        - fqs_ts:
            Number of time steps per second in output representation
        - n_ts:
            Number of time steps in a bar, 2**`precision`

    .. note:: Obsolete
    """
    def __init__(self, fl_nm, precision=5):
        """
        :param fl_nm: Path & MIDI file name
        :param precision: The smallest music note duration is 1/`2^precision`-th
        """
        self.fnm = fl_nm
        self.precision = precision
        self.mu = MidoUtil()
        self.pmu = PrettyMidiUtil()

        self.mf = MidiFile(self.fnm)
        self.pm = PrettyMIDI(self.fnm)

        msgs = self.mu.get_msgs_by_type(self.mf, 'time_signature')
        # ic(msgs)
        assert len(msgs) == 1
        msg = msgs[0]
        self.beat_per_bar = msg.numerator
        self.frac_per_beat = msg.denominator
        assert math.log2(self.frac_per_beat).is_integer()
        # ic(self.beat_per_bar, self.frac_pow_per_beat)

        tempos = self.mu.get_tempo_changes(self.mf)
        assert len(tempos) == 1
        self.tempo = tempos[0]
        self.bpm = tempo2bpm(self.tempo)

        spb = self.bpm / 60  # Second per beat
        spt = spb / (2 ** self.precision / self.frac_per_beat)  # Second that each time slot consumes
        # ic(spt)
        self.fqs_ts = 1/spt
        self.n_ts = 2**self.precision * self.beat_per_bar / self.frac_per_beat

    def __call__(self, method='bwmp'):
        d_f = dict(
            bwmp=self.bar_with_max_pitch
        )
        return d_f[method]

    def bar_with_max_pitch(self):
        """
        For each bar, pick the track with the highest average pitch

        If multiple notes at a time step, pick the one with the highest pitch
        """
        # ic(self.precision - self.frac_per_beat, (self.precision - self.frac_per_beat)**2)
        # ic(spb, spt, 1/spt)
        # pr = self.pm.get_piano_roll(fs=)
        # ic(self.pm.instruments[0].notes[:10])
        # ic(pr, pr.shape)
        self.pmu.plot_piano_roll(self.pm, fqs=self.fqs_ts)


class MxlMelodyExtractor:
    """
    Given MXL file, export single-track melody representations,
    as pitch encodings with bar separations or as MXL files

    Each bar is divided into equidistant slots of music note length, given by a `prec` attribute for precision
        e.g. precision <- 5 means
    The number of slots for a bar hence depends on the time signature

    For now, enforce that at each time step, there will be only one note played
    In case of multiple notes at a time step,
    the concurrent notes are filtered such that only the note with the highest pitch remains

    The pitch will be encoded as integer in [0-127] per MIDI convention
        Triplets are handled with a special id at the last quarter
    """

    def __init__(self, fl_nm, precision=5, n=None, verbose=True):
        """
        :param fl_nm: Music MXL file path
        :param precision: Time slot duration as negative exponent of 2
        :param n: If specified, read up until n-th element for each part
        """
        self.fnm = fl_nm
        self.prec = precision
        self.verbose = verbose

        self.scr: m21.stream.Score = m21.converter.parse(self.fnm)
        if n is not None:
            for p in self.scr.parts:
                p.remove(list(p)[n:])

        lens = [len(p[m21.stream.Measure]) for p in self.scr.parts]
        assert_list_same_elms(lens)
        self.bar_strt_idx = None  # First element in a `Part` can be a non-measure

        pnms = [p.partName for p in self.scr.parts]
        if not len(pnms) == len(set(pnms)):  # Unique part names
            for idx, p in enumerate(self.scr.parts):
                p.partName = f'{p.partName}, CH #{idx+1}'

        # The **first** bar of at least one track should contain the tempo & the time signature
        bars_1st = [next(p[m21.stream.Measure]) for p in self.scr.parts]  # Can be in a drum track
        tempos_strt = [bar[m21.tempo.MetronomeMark] for bar in bars_1st]
        assert any(len(tempo) > 0 for tempo in tempos_strt)
        self.tempo_strt = next(filter(lambda tempos: len(tempos), tempos_strt))[0]

        tss_strt = [bar[m21.meter.TimeSignature] for bar in bars_1st]
        assert any(len(time_sigs) > 0 for time_sigs in tss_strt)

        # Remove drum tracks
        def is_drum(part):
            """
            :return: True if `part` contains *only* `Unpitched`
            """
            return list(part[m21.note.Unpitched]) and not list(part[m21.note.Note])
        parts_drum = filter(lambda p_: any(p_[drum] for drum in [
            m21.instrument.BassDrum,
            m21.instrument.BongoDrums,
            m21.instrument.CongaDrum,
            m21.instrument.SnareDrum,
            m21.instrument.SteelDrum,
            m21.instrument.TenorDrum,
        ]) or is_drum(p_), self.scr.parts)
        for p in parts_drum:
            self.scr.remove(p)

        # The tempo available in a drum channel & was removed
        if not any(len(next(p[m21.stream.Measure])[m21.tempo.MetronomeMark]) > 0 for p in self.scr.parts):
            next(self.scr.parts[0][m21.stream.Measure]).insert(self.tempo_strt)

        self._vertical_bars: list[MxlMelodyExtractor.VerticalBar] = []

        self.tokenizer = MxlMelodyExtractor.Tokenizer(self.prec)

    def beyond_precision(self):
        """
        :return: True if the music score contains notes of too short duration relative to precision
        """
        min_dur = 2**2 / 2**self.prec  # Duration in quarterLength
        return any(e.duration.quarterLength < min_dur for e in it_m21_elm(self.scr))

    class VerticalBar:
        """
        Contains all the bars for each channel in the score, at the same time
        """
        def __init__(self, bars: dict[str, m21.stream.Measure]):
            self.bars = bars
            nums = [bar.number == 0 for bar in bars.values()]
            assert_list_same_elms(nums)
            self.n = nums[0]

            tss = [b[m21.meter.TimeSignature] for b in self.bars.values()]
            self._time_sig = None
            if any(tss):
                assert all(len(t) == 1 for t in tss)  # At most 1 time signature per bar
                tss = [next(t) for t in tss]
                dss = [(ds.numerator, ds.denominator) for ds in tss]  # Time signature across bars should be the same
                assert_list_same_elms(dss)

                self._time_sig = dss[0]

            tempos = [b[m21.tempo.MetronomeMark] for b in self.bars.values()]
            self._tempo = None
            if any(tempos):
                tempos = [t for t in tempos if len(t) != 0]
                # When multiple tempos, take the mean
                tempos = [m21.tempo.MetronomeMark(number=np.array([t.number for t in ts]).mean()) for ts in tempos]
                bpms = [t.number for t in tempos]
                assert_list_same_elms(bpms)

                self._tempo = m21.tempo.MetronomeMark(number=bpms[0])

        def __getitem__(self, key):
            return self.bars[key]

        def __iter__(self):
            return iter(self.bars.keys())

        # def __del__(self, key: Union[str, list[str]]):
        #     if not isinstance(key, list):
        #         key = [key]
        #     for k in key:
        #         del self.bars[k]

        def values(self):
            return self.bars.values()

        def items(self):
            return self.bars.items()

        @property
        def time_sig(self):
            return self._time_sig

        @time_sig.setter
        def time_sig(self, val):
            self._time_sig = val

        @property
        def tempo(self):
            return self._tempo

        @tempo.setter
        def tempo(self, val):
            self._tempo = val

        def n_slot(self, prec):
            """
            Per the symbolic representation

            :param prec: Precision
            :return: Number of slots in a bar
            """
            numer, denom = self.time_sig
            n_slot = 2**prec * numer / denom
            if not n_slot.is_integer():
                warn(f'Number of slot per bar not an integer for '
                     f'precision [{prec}] and time signature [{self.time_sig}]')
            return int(n_slot)

        def single(self, inplace=False):
            """
            For each time step in each bar, filter notes such that only the note with the highest pitch remains

            Chords are effectively converted to notes

            .. note:: The `bars` attribute is modified
            """
            obj = self if inplace else deepcopy(self)
            del_pnms = []
            for pnm, bar in obj.bars.items():
                # ic(bar.number)
                if bar.hasVoices():
                    # Seems to be a problem in prior processing: overlapping notes => Fix it
                    clefs = [e for e in bar if isinstance(e, m21.clef.Clef)]
                    ln_clef = len(clefs)

                    if ln_clef >= 1:
                        warn(f'Clef found in bar {bar.number}, channel [{pnm}] containing voices '
                             f'- voice durations potentially adjusted')
                        if any(sum(e.duration.quarterLength for e in
                                   it_m21_elm(v, types=(m21.note.Note, m21.note.Rest, m21.chord.Chord)))
                               != v.duration.quarterLength
                               for v in bar.voices):
                            offset_prev = 0
                            for clef in clefs:
                                for v in bar.voices:
                                    if sum(e.duration.quarterLength for e in
                                           it_m21_elm(v, types=(m21.note.Note, m21.note.Rest, m21.chord.Chord))
                                           ) != v.duration.quarterLength:  # Inconsistent
                                        notes_before = [e for e in v if offset_prev <= e.offset < clef.offset]
                                        l_n = len(notes_before)
                                        if l_n > 0:
                                            n = notes_before[-1]
                                            # Up until start of Clef
                                            notes_before_ = [e for e in v if e.offset < clef.offset]  # ALl notes
                                            dur_prev = 0 if l_n == 1 else sum(
                                                e.duration.quarterLength for e in notes_before_[:-1]
                                            )
                                            n.duration = m21.duration.Duration(quarterLength=clef.offset - dur_prev)
                                offset_prev = clef.offset
                    voices = bar.voices
                    if '1' in [v.id for v in voices]:
                        soprano = next(filter(lambda v: v.id == '1', voices))

                        # Insert notes in `soprano` by offset
                        obj.bars[pnm].remove(list(voices))
                        obj.bars[pnm].insert(flatten([e.offset, e] for e in soprano))
                    else:
                        del_pnms.append(pnm)
                        warn(f'Soprano not found in bar [{bar.number}], channel [{pnm}] - Bar removed')
            for pnm in del_pnms:
                del obj.bars[pnm]
            for pnm, bar in obj.bars.items():
                assert not bar.hasVoices()
                notes = bar.notes  # TODO: include rest
                assert notes.isSorted
                assert_notes_no_overlap(notes)

                def chord2note(c):
                    return max(c.notes, key=lambda n: n.pitch.frequency)
                for note in notes:
                    if isinstance(note, m21.chord.Chord):
                        obj.bars[pnm].replace(note, chord2note(note))
            return obj

        def avg_pitch(self, method='fqs', val_rest=0):
            """
            :param method: Mapping function from Note to value,
                either `fqs` for pitch frequency, or `midi` for MIDI pitch encoding
            :param val_rest: The value assigned to `Rest`
            :return: Average pitch, weighted by duration, for each bar
            :precondition: Each bar contains `music21.note.Note`s only

            Pitch value by either MIDI representation integer, or frequency
            """
            d_func = dict(
                fqs=lambda n: n.pitch.frequency,
                midi=lambda n: n.pitch.midi
            )
            f = d_func[method]

            def _avg_pitch(b):
                if b.notes:
                    fs = [f(n) for n in b.notes]
                    ws = [n.duration.quarterLength for n in b.notes]
                    fs.append(val_rest)
                    ws.append(sum([r.quarterLength for r in b[m21.note.Rest]]))
                    return np.average(np.array(fs), weights=np.array(ws))
                else:
                    return float('-inf')
            return {pnm: _avg_pitch(bar) for pnm, bar in self.items()}

        def pnm_with_max_pitch(self, method='fqs', val_rest=0):
            """
            :return: The part name key, that has the maximum pitch, per `avg_pitch`
            """
            for pnm, b in self.bars.items():
                # ic(pnm, b)
                assert not b[m21.note.Unpitched]
            pchs = self.avg_pitch(method=method, val_rest=val_rest)
            return max(self.bars, key=lambda p: pchs[p])

    def vertical_bars(self, scr) -> list['MxlMelodyExtractor.VerticalBar']:
        """
        :return: List of `VerticalBar`s
        """
        d_bars = {p.partName: list(sorted(p[m21.stream.Measure], key=lambda b: b.number)) for p in scr.parts}
        self.bar_strt_idx = list(d_bars.values())[0][0].number
        # All parts starts with the same bar number
        assert all(bars[0].number == self.bar_strt_idx for bars in d_bars.values())

        vbs = [self.VerticalBar({b.activeSite.partName: b for b in bars}) for bars in zip(*d_bars.values())]
        ts = vbs[0].time_sig  # Get Score starting time signature
        tp = vbs[0].tempo
        assert ts is not None and tp is not None
        for vb in vbs:  # Unroll time signature to each VerticalBar
            if vb.time_sig is None:
                vb.time_sig = ts
            else:
                ts = vb.time_sig

            if vb.tempo is None:
                vb.tempo = tp
            else:
                tp = vb.tempo
        return vbs

    @staticmethod
    def score_seconds(vert_bars, as_str=True):
        """
        :param vert_bars: `VerticalBar`s for the score
        :param as_str: If true, a human-readable string representation is returned
        """
        s = int(sum(vb.tempo.durationToSeconds(next(iter(vb.bars.values())).duration) for vb in vert_bars))
        return sec2mmss(s) if as_str else s

    @property
    def mean_tempo(self):
        bars_with_tempo = {
            p.partName: list(filter(lambda b: b[m21.tempo.MetronomeMark], self.scr.parts[i][m21.stream.Measure]))
            for i, p in enumerate(self.scr.parts)
        }
        dur = self.scr.duration.quarterLength

        def _mean_tempo(bars):
            tempos_by_bar = [np.array([t.number for t in bar[m21.tempo.MetronomeMark]]).mean() for bar in bars]
            n = len(bars)
            tempos_with_dur = [
                [t, (bars[idx+1].offset if idx+1 < n else dur) - bar.offset]
                for idx, (t, bar) in enumerate(zip(tempos_by_bar, bars))
            ]
            return tempos_with_dur
        tempos, weights = zip(*flatten(_mean_tempo(bars) for bars in bars_with_tempo.values()))
        return np.average(np.array(tempos), weights=np.array(weights))

    class Tokenizer:
        """
        Handles pitch id encoding of each music21.stream.Measure

        Enforce that each time slot should be assigned an id, only once
        """
        class Slot:
            def __init__(self):
                self._id = nan
                self.set = False

            @property
            def id(self):
                return self._id

            @id.setter
            def id(self, val):
                assert not self.set  # Should be defined only once
                self.set = True
                self._id = val

            def __repr__(self):
                return f'<{self.__class__.__qualname__} id={self._id} set={self.set}>'

        class _Tokenizer:
            """
            Encodes **single** music21 element in a bar to the pitch id

            Expects single `Note` or `Rest`
            """
            D_CONF = config('Melody-Extraction.tokenizer')

            def __init__(self):
                self.n_spec = self.D_CONF['n_special_token']
                self.spec = self.D_CONF['vocab_special']
                self.enc = self.D_CONF['encoder']
                self.dec = self.D_CONF['decoder']

            def __call__(self, obj):
                if isinstance(obj, m21.note.Note):
                    return self.enc[obj.pitch.midi]
                elif isinstance(obj, m21.note.Rest):
                    return self.enc['[REST]']
                elif isinstance(obj, str) and obj in self.spec:
                    return self.enc[obj]
                else:
                    raise ValueError(f'Unexpected object: {obj}')

            def decode(self, obj) -> Union[int, np.integer, str]:
                return self.dec[obj]

        class Token:
            def __init__(self, tok: Union[int, np.integer, str], duration):
                """
                :param tok: A pitch or a special token
                :param duration: Duration in quarterLength
                """
                self.dur = duration
                dur = m21.duration.Duration(quarterLength=duration)
                if isinstance(tok, str):
                    if tok == '[REST]':
                        self.note = m21.note.Rest(duration=dur)
                    else:
                        ic(tok)
                        exit(1)
                else:  # Integer
                    self.note = m21.note.Note(tok, duration=dur)

            def __repr__(self):
                return f'<{self.__class__.__qualname__} note={self.note} dur={self.dur}>'

        def __init__(self, precision: int):
            self.prec = precision

            self.tokenizer = MxlMelodyExtractor.Tokenizer._Tokenizer()
            self.enc_trip = self.tokenizer('[TRIP]')

            self.time_sigs = None

        def __call__(
                self,
                bars: Union[
                    Union[
                        m21.stream.Measure,
                        list[m21.stream.Measure],
                        list[tuple[m21.stream.Measure, m21.meter.TimeSignature]]
                    ],
                    Union[
                        m21.note.Note,
                        m21.note.Rest,
                        str
                    ]
                ],
                time_sigs: Union[m21.meter.TimeSignature, list[m21.meter.TimeSignature]] = None
        ):
            """
            :param bars: Bar, List of bars, or single element (see `_Tokenizer`)
            :param time_sigs: List of time_sigs
                If unspecified, expect `bars` to contain list of 2-tuple of (bar, time_sig)
            :return: Encoded pitch ids
            """
            if isinstance(bars, (m21.note.Note, m21.note.Rest, str)):
                return self.tokenizer(bars)

            if time_sigs is None:
                lst_bar_n_ts = bars
                self.time_sigs = [ts for _, ts in lst_bar_n_ts]
            else:
                assert len(bars) == len(time_sigs)
                lst_bar_n_ts = list(zip(bars, time_sigs))
                self.time_sigs = time_sigs

            def _call(bar, time_sig):
                n_slots_per_beat, n_slots = time_sig2n_slots(time_sig, self.prec)
                enc = [MxlMelodyExtractor.Tokenizer.Slot() for _ in range(n_slots)]
                # ic(bar.number)
                r_dur = time_sig2ratio(time_sig)
                for e in group_triplets(bar):
                    if isinstance(e, list):  # Triplet case
                        lst = e
                        dur = sum(e.duration.quarterLength for e in lst)  # Over `Fraction`s
                        # The smallest duration of triplet that can be encoded is 4 time slots
                        num_ea = (dur / 4 * n_slots_per_beat) * r_dur
                        assert num_ea.is_integer()
                        strt_idx = lst[0].offset * n_slots_per_beat
                        assert strt_idx.is_integer()
                        # Special encoding for triplets at the end
                        for offset, elm in zip([0, 1, 2, 3], lst + ['[TRIP]']):
                            idxs = (strt_idx + np.arange(num_ea) + offset * num_ea).astype(int)
                            id_ = self.tokenizer(elm)
                            for idx in idxs:
                                enc[idx].id = id_
                    else:
                        strt, dur = e.offset, e.duration.quarterLength
                        num = n_slots_per_beat*dur * r_dur
                        assert num.is_integer()
                        idxs_ = (np.linspace(strt, strt+dur, num=int(num)+1) * n_slots_per_beat * r_dur)[:-1]
                        idxs = idxs_.astype(int)
                        assert np.all(idxs_-idxs == 0)  # Should be integers

                        id_ = self.tokenizer(e)
                        for idx in idxs:
                            enc[idx].id = id_
                assert all(s.set for s in enc)  # Each slot has an id set
                return [s.id for s in enc]

            ids = [_call(*args) for args in lst_bar_n_ts]
            id_bar = self.tokenizer('[SEP]')
            return reduce(lambda a, b: a+[id_bar]+b, ids)  # Join the encodings with bar separation

        def decode(
                self,
                ids: list[int],
                time_sigs: Union[list[m21.meter.TimeSignature], str] = None
        ) -> list[m21.stream.Measure]:
            """
            :param ids: A list of token ids
            :param time_sigs: Time Signatures of each bar
                If 'infer`, inferred from time slots
                If not given, the one from last encoding call is used
            :return: A human-readable representation of `ids`
            """
            ids = np.asarray(ids)
            enc_sep = self.tokenizer('[SEP]')
            idxs = np.where(ids == enc_sep)[0]
            lst_ids = np.split(ids, idxs)
            # All lists except the 1st one starts with the bar SEP encoding
            lst_ids = [(l if idx == 0 else l[1:]) for idx, l in enumerate(lst_ids)]

            if time_sigs == 'infer':
                denom = 4  # Assumption without loss of generality
                n_slots_per_beat = time_sigs = (1/denom / (2 ** -self.prec))
                numers = [len(ids_) / n_slots_per_beat for ids_ in lst_ids]
                assert all(n.is_integer() for n in numers)
                time_sigs = [m21.meter.TimeSignature(f'{int(n)}/{denom}') for n in numers]
            elif time_sigs is None:
                time_sigs = self.time_sigs

            assert len(lst_ids) == len(time_sigs)
            # Each element as 2-tuple of (id list, count list)
            lst_ids_n_cts = [tuple(zip(*compress(list(l)))) for l in lst_ids]
            bars = [
                self._decode(ids_n_cnt, time_sig, number=i)
                for i, (ids_n_cnt, time_sig) in enumerate(zip(lst_ids_n_cts, time_sigs))
            ]

            # Add time signatures
            it_b, it_ts = iter(bars), iter(time_sigs)
            ts = next(it_ts)
            next(it_b).insert(ts)
            for bar, ts_ in zip(it_b, it_ts):
                if not (ts_.denominator == ts.denominator and ts_.numerator == ts.numerator):
                    ts = ts_
                    bar.insert(ts)
            return bars

        def _decode(
                self,
                ids_n_cnt: tuple[tuple[int], tuple[int]],
                time_sig: m21.meter.TimeSignature,
                number=None
        ):
            """
            For single bar
            """
            # ic(number)
            kwargs = {} if number is None else dict(number=number)
            bar = m21.stream.Measure(**kwargs)
            n_slots_per_beat, n_slots = time_sig2n_slots(time_sig, self.prec)
            r_dur = time_sig2ratio(time_sig)
            ids_, counts = ids_n_cnt
            durs = [count/n_slots_per_beat / r_dur for count in counts]

            def id_n_dur2tok(i_, d_):
                return MxlMelodyExtractor.Tokenizer.Token(self.tokenizer.decode(i_), d_).note

            def get_toks():
                ids__ = np.asarray(ids_)
                # Indices for each triplet
                idxs_trip_end = np.where(ids__ == self.enc_trip)[0]

                idxs_trip_strt = []
                for idx_end in idxs_trip_end:
                    dur = durs[idx_end]
                    idx_strt = idx_end-1
                    dur_total = durs[idx_strt]
                    # The starting index for each triplet should start from 0 and is normally 3 indices away
                    # but edge case
                    while dur_total < dur * 3:
                        idx_strt -= 1
                        dur_total += durs[idx_strt]
                    # `dur_total` may be greater than `dur * 3`, if the previous note has the same pitch
                    idxs_trip_strt.append(idx_strt)

                if idxs_trip_end.size >= 1:
                    l = ids__.size
                    idx = 0
                    lst_tok = []
                    while idx < l:
                        if idx in idxs_trip_strt:
                            # idx_end = idxs_trip_end[np_index(idxs_trip_strt, idx)]
                            idx_end = idxs_trip_end[idxs_trip_strt.index(idx)]
                            dur_total = durs[idx_end] * 4  # The triplet encoding
                            # Sanity check of encoding precision
                            ratio = n_slots_per_beat / 4
                            assert ratio.is_integer()
                            assert (dur_total * ratio).is_integer()  # 4 for duration in quarterLength
                            dur = Fraction(int(dur_total * ratio), int(3 * ratio))

                            dur_non_trip = sum(durs[idx:idx_end]) + dur_total / 4 - dur_total
                            if dur_non_trip != 0:  # 1st triplet pitch same as prior normal pitch
                                lst_tok.append(id_n_dur2tok(ids__[idx], dur_non_trip))  # The Non-triplet note
                                dur_total -= dur_non_trip

                            ln = idx_end - idx
                            if ln == 3:  # Normal case
                                for id__ in ids__[idx:idx+3]:
                                    lst_tok.append(id_n_dur2tok(id__, dur))
                            else:  # Multiple contiguous triplet pitches with the same pitch
                                if ln == 1:  # Same pitch for all 3 triplet notes
                                    for i in range(3):
                                        lst_tok.append(id_n_dur2tok(ids__[idx], dur))
                                elif ln == 2:
                                    dur1st, dur2nd = durs[idx], durs[idx+1]
                                    lst_tok.append(id_n_dur2tok(ids__[idx], dur))
                                    if dur1st == dur2nd * 2:
                                        lst_tok.append(id_n_dur2tok(ids__[idx], dur))
                                    else:
                                        assert dur1st*2 == dur2nd
                                        lst_tok.append(id_n_dur2tok(ids__[idx+1], dur))
                                    lst_tok.append(id_n_dur2tok(ids__[idx+1], dur))
                            idx = idx_end+1
                        else:
                            lst_tok.append(id_n_dur2tok(ids__[idx], durs[idx]))
                            idx += 1
                    return lst_tok
                else:
                    return [id_n_dur2tok(i_, d_) for i_, d_ in zip(ids__, durs)]
            bar.append(get_toks())

            # For quarterLength in music21
            dur_bar = time_sig.numerator * (4 / time_sig.denominator)
            if bar.duration.quarterLength != dur_bar:
                ic(number, dur_bar, bar.duration.quarterLength)
                for e in bar:
                    ic(e, e.offset, e.duration)
            assert bar.duration.quarterLength == dur_bar
            return bar

    @property
    def score_title(self):
        title = self.scr.metadata.title
        if title.endswith('.mxl'):
            title = title[:-4]
        return title

    def bar_with_max_pitch(self, exp=None):
        """
        :param exp: Export format,
            `mxl`: write to MXL file
            `symbol`: symbolic encodings, pitch ids of each time slot
            otherwise, return the `music21.stream.Score` object
        For each bar, pick the track with the highest average pitch
        """
        scr = deepcopy(self.scr)
        composer = PKG_NM
        scr.metadata.composer = composer

        # Pick a `Part` to replace elements one by one, the 1st part selected as it contains all metadata
        idx_part = 0
        scr.remove(list(filter(lambda p: p is not scr.parts[idx_part], scr.parts)))
        assert len(scr.parts) == 1
        part = scr.parts[0]
        pnm = part.partName

        def pnm_ori(nm):
            return nm[:nm.rfind(', CH #')]
        pnm_ori_ = pnm_ori(pnm)
        for p in self.scr.parts[1:]:
            # There should be no tempo in all other channels, unless essentially the "same" channel
            assert len(p[m21.tempo.MetronomeMark]) == 0 or pnm_ori(p.partName) == pnm_ori_

        vbs = self.vertical_bars(self.scr)
        if self.verbose:
            print(f'{now()}| Extracting music [{stem(self.fnm)}] of duration [{self.score_seconds(vbs)}]... ')
        for idx, bar in enumerate(part[m21.stream.Measure]):  # Ensure each bar is set
            vb = vbs[idx].single()
            pnm_ = vb.pnm_with_max_pitch(method='fqs')
            assert bar.number == idx + self.bar_strt_idx
            assert part.index(bar) == idx+1
            part.replace(bar, vb.bars[pnm_])

        # Set instrument as Piano
        instr = m21.instrument.Piano()
        [part.remove(ins) for ins in part[m21.instrument.Instrument]]
        part.insert(instr)
        part.partName = f'{PKG_NM}, {instr.instrumentName}, CH #1'

        # Set tempo
        [bar.removeByClass(m21.tempo.MetronomeMark) for bar in part[m21.stream.Measure]]
        self.tempo_strt.number = self.mean_tempo
        bar0 = part.measure(self.bar_strt_idx)
        bar0.insert(self.tempo_strt)

        title = f'{self.score_title}, bar with max pitch'
        if exp == 'mxl':
            dir_nm = config(f'{DIR_DSET}.MXL_EG.dir_nm')
            dir_nm = f'{dir_nm}_out'
            scr.write(fmt='mxl', fp=os.path.join(PATH_BASE, DIR_DSET, dir_nm, f'{title}.mxl'))
        elif exp == 'symbol':
            # Get time signature for each bar
            lst_bar_n_ts = bars2lst_bar_n_ts(part[m21.stream.Measure])
            return self.tokenizer(lst_bar_n_ts)
        else:
            return scr

    def slot_with_max_pitch(self):
        """
        For each time slot, pick track with the highest pitch
        """
        pass

    def encoding2score(self, ids, save=False):
        bars = self.tokenizer.decode(ids, time_sigs='infer')
        # Set default tempo
        bars[0].insert(m21.tempo.MetronomeMark(number=config('Melody-Extraction.output.BPM')))
        part = m21.stream.Part()
        instr = m21.instrument.Piano()
        bars[0].insert(m21.meter.TimeSignature('12/8'))
        part.partName = f'{PKG_NM}, {instr.instrumentName}, CH #1'
        part.append(instr)
        part.append(bars)

        scr = m21.stream.Score()
        scr.append(m21.metadata.Metadata())
        scr.metadata.composer = PKG_NM
        scr.metadata.title = f'{self.score_title}, decoded'
        scr.append(part)

        if save:
            dir_nm = config(f'{DIR_DSET}.MXL_EG.dir_nm')
            dir_nm = f'{dir_nm}_out'
            scr.write(fmt='mxl', fp=os.path.join(PATH_BASE, DIR_DSET, dir_nm, f'{scr.metadata.title}.mxl'))
        else:
            scr.show()
        return scr


class MelodyTokenizer:
    """
    Music MXL file to melody pitch representation tokenizer

    Wrapper for `MxlMelodyExtractor` (TODO)
    """

    D_CONF = config('Melody-Extraction.tokenizer')

    MAP_DF = {
        '[SEP]': '<s>',
        '[TRIP]': '<t>',
        '[PAD]': '<p>',
        '[REST]': '<r>'
    }

    def __init__(self, spec_map: Callable[[str], str] = MAP_DF.get):
        """
        :param spec_map: Maps each decoded special token to a new string
        """
        self.n_spec = self.D_CONF['n_special_token']
        self.spec = self.D_CONF['vocab_special']
        self.enc = self.D_CONF['encoder']
        self.dec = self.D_CONF['decoder']
        self.spec_map = spec_map
        self.pchs = set(range(2**7))  # Valid pitch encodings per MIDI

        self.vocab = [self.id2str(id_) for id_ in self.dec]

    def id2str(self, id_: Union[int, str]) -> str:
        """
        :param id_: Encoded id
        :return: A string representation
        """
        id_ = self.dec[id_]
        if id_ in self.pchs:
            return m21.pitch.Pitch(midi=id_).nameWithOctave
        else:
            assert isinstance(id_, str)
            return self.spec_map(id_) if self.spec_map is not None else id_

    def decode(
            self,
            ids: Union[int, list[int], list[list[int]], np.ndarray, list[np.ndarray]],
            return_joined=True
    ) -> Union[str, list[str], list[list[str]]]:
        """
        :param ids: Pitch ids
        :param return_joined: If True and iterable ids passed in, the melody is joined into a single string
        :return: A string representation of each id
        """

        def _decode(ids_: list[int]):
            return ' '.join(self.id2str(id_) for id_ in ids_) if return_joined else [self.id2str(id_) for id_ in ids_]
        if isinstance(ids, int):
            return self.id2str(ids)
        elif isinstance(ids, list) and isinstance(ids[0], (list, np.ndarray)):
            return list(conc_map(_decode, ids))
        else:
            assert isinstance(ids, (list, np.ndarray))
            return _decode(ids)


def extract(dnms: list[str], exp='json') -> list[dict[str]]:
    """
    :param dnms: Dataset names
    :param exp: Encoded songs export format
        `json` for JSON file
    """
    count = 0
    count_suc = 0
    songs = []
    fnms = {dnm: fl_nms(dnm, k='song_fmt_exp') for dnm in dnms}
    n_songs = sum(len(e) for e in fnms.values())
    n = len(str(n_songs))
    for dnm, fnms in fnms.items():
        for fnm in fnms:
            fnm_ = stem(fnm)
            num = f'{{:>0{n}}}'.format(count)
            log(f'Encoding song #{log_s(num, c="i")} [{log_s(fnm_, c="i")}]... ')
            me = MxlMelodyExtractor(fnm, verbose=False)
            if has_quintuplet(me.scr):
                warn(f'Song [{fnm_}] ignored for containing quintuplets')
            elif invalid_triplets(me.scr):
                warn(f'Song [{fnm_}] ignored for containing invalid triplets')
            elif me.beyond_precision():  # TODO: resolve later
                warn(f'Song [{fnm_}] ignored for duration beyond precision')
            else:
                ids = me.bar_with_max_pitch(exp='symbol')
                num_ = f'{{:>0{n}}}'.format(count_suc)
                log(f'Encoding song [{fnm_}] success #{num_}', c='g')
                songs.append(dict(
                    nm=fnm_,
                    ids=ids
                ))
                count_suc += 1
            count += 1
    log(f'{count_suc} songs encoded', c='g')
    if exp == 'json':
        fnm = 'Song-ids'
        with open(os.path.join(PATH_BASE, DIR_DSET, config(f'{DIR_DSET}.my.dir_nm'), f'{fnm}.json'), 'w') as f:
            json.dump(songs, f, indent=4)
    return songs


if __name__ == '__main__':
    from icecream import ic

    def check_midi():
        fnm = eg_songs('Shape of You')
        # fnm = eg_songs('My Favorite Things')
        me = MidiMelodyExtractor(fnm)
        ic(me.bpm)
        me.bar_with_max_pitch()
    # check_midi()

    def check_mxl():
        # fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        fnm = eg_songs('Shape of You', fmt='MXL')
        ic(fnm)
        me = MxlMelodyExtractor(fnm)
        me.bar_with_max_pitch(exp='mxl')
    # check_mxl()

    def extract_encoding():
        # fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        fnm = eg_songs('Shape of You', fmt='MXL')
        ic(fnm)
        me = MxlMelodyExtractor(fnm, n=None)
        me.bar_with_max_pitch(exp='symbol')
    # extract_encoding()

    def sanity_check_encoding():
        # fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        # fnm = eg_songs('Shape of You', fmt='MXL')
        fnm = eg_songs('平凡之路', fmt='MXL')
        ic(fnm)
        me = MxlMelodyExtractor(fnm, n=None)
        ids = me.bar_with_max_pitch(exp='symbol')
        ic(ids[:2**5])
        ic(me.encoding2score(ids, save=True))
    # sanity_check_encoding()

    # ic(fl_nms('LMD_Cleaned', k='song_fmt_exp'))

    def encode_a_few():
        # n = 2**6
        dnm = 'POP909'
        fnms = fl_nms(dnm, k='song_fmt_exp')
        for idx, fnm in enumerate(fnms[66+136+289:]):
        # for idx, fnm in enumerate(fnms):
            ic(idx, stem(fnm))
            me = MxlMelodyExtractor(fnm)
            if has_quintuplet(me.scr):
                warn(f'Song [{stem(fnm)}] ignored for containing quintuplets')
            elif invalid_triplets(me.scr):
                warn(f'Song [{stem(fnm)}] ignored for containing invalid triplets')
            elif me.beyond_precision():  # TODO: resolve later
                warn(f'Song [{stem(fnm)}] ignored for duration beyond precision')
            else:
                ids = me.bar_with_max_pitch(exp='symbol')
                print(f'Encoding song [{stem(fnm)}] success')
                me.encoding2score(ids, save=True)
            # exit(1)
    # encode_a_few()

    def store_encoding():
        dnms = ['LMD_Cleaned', 'POP909']
        extract(dnms)
    # store_encoding()

    def check_encoding_export():
        fnm = 'Song-ids'
        with open(os.path.join(config('path-export'), f'{fnm}.json'), 'r') as f:
            songs = json.load(f)
            ic(len(songs), songs[0])
    # check_encoding_export()

    def check_melody_tokenizer():
        from musicnlp.model import Loader

        mt = MelodyTokenizer()
        ic(len(mt.vocab), mt.vocab[:20])
        ml = Loader()
        ids = list(ml)
        # ic(ids)
        ic(mt.decode(ids[0]))
        # ic(mt.decode(ids))
    # check_melody_tokenizer()

    # ic(__name__, __file__)


