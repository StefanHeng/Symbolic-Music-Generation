import os
import math
from copy import deepcopy
from warnings import warn
from typing import Union
from fractions import Fraction

from util import *

import numpy as np
from mido import MidiFile
import pretty_midi
from pretty_midi import PrettyMIDI
import music21 as m21


def assert_list_same_elms(lst):
    assert all(l == lst[0] for l in lst)


def assert_notes_no_overlap(notes: list[Union[m21.note.Note, m21.chord.Chord]]):
    """
    Asserts that the notes don't overlap, given the start time and duration
    """
    if len(notes) >= 2:
        end = notes[0].offset + notes[0].duration.quarterLength
        for note in notes[1:]:
            # Current note should begin, after the previous one ends
            # Since numeric representation of one-third durations, aka tuplets
            assert end <= note.offset or math.isclose(end, note.offset, abs_tol=1e-6)
            end = note.offset + note.duration.quarterLength


def it_bar_elm(bar, types=(m21.note.Note, m21.note.Rest)):
    """
    Iterates elements in a bar, for those that are instances of that of `type`, in the original order
    """
    return iter(filter(lambda elm: isinstance(elm, types), bar))


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


def group_triplets(bar) -> list[Union[list[m21.note.Note], Union[m21.note.Note, m21.note.Rest]]]:
    """
    Identify triplets from a bar from normal notes & group them
    """
    lst = []
    it = it_bar_elm(bar)
    elm = next(it, None)
    while elm:
        if 'Triplet' in elm.fullName:
            elm2, elm3 = next(it, None), next(it, None)
            assert elm2 is not None and elm3 is not None \
                   and 'Triplet' in elm2.fullName and 'Triplet' in elm3.fullName
            lst.append([elm, elm2, elm3])
        else:
            lst.append(elm)
        elm = next(it, None)
    return lst


def time_sig2n_slots(time_sig, precision):
    """
    :return: 2 tuple of (#time slots per beat, #time slots in total)
    """
    denom = time_sig.denominator
    numer = time_sig.numerator
    n_slots_per_beat = (1 / denom / (2 ** -precision))
    assert n_slots_per_beat.is_integer()
    n_slots = int(numer * n_slots_per_beat)
    return n_slots_per_beat, n_slots


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

    def __init__(self, fl_nm, precision=5, n=None):
        """
        :param fl_nm: Music MXL file path
        :param precision: Time slot duration as negative exponent of 2
        :param n: If specified, read up until n-th element for each part
        """
        self.fnm = fl_nm
        self.prec = precision

        self.scr: m21.stream.Score = m21.converter.parse(self.fnm)
        if n is not None:
            for p in self.scr.parts:
                p.remove(list(p)[n:])

        ic(self.scr.seconds)  # TODO: MXL file duration in physical time
        lens = [len(p[m21.stream.Measure]) for p in self.scr.parts]
        assert_list_same_elms(lens)
        self.bar_strt_idx = None  # First element in a `Part` can be a non-measure

        pnms = [p.partName for p in self.scr.parts]
        if not len(pnms) == len(set(pnms)):  # Unique part names
            for idx, p in enumerate(self.scr.parts):
                p.partName = f'{p.partName}, CH #{idx+1}'

        # Remove drum tracks
        parts_drum = filter(lambda p_: any(p_[drum] for drum in [
            m21.instrument.BassDrum,
            m21.instrument.BongoDrums,
            m21.instrument.CongaDrum,
            m21.instrument.SnareDrum,
            m21.instrument.SteelDrum,
            m21.instrument.TenorDrum,
        ]), self.scr.parts)
        for p in parts_drum:
            self.scr.remove(p)

        # The **first** bar of at least one track should contain the tempo & the time signature
        bars_1st = [next(p[m21.stream.Measure]) for p in self.scr.parts]
        tempos_strt = [bar[m21.tempo.MetronomeMark] for bar in bars_1st]
        assert any(len(tempo) > 0 for tempo in tempos_strt)
        self.tempo_strt = next(filter(lambda tempos: len(tempos), tempos_strt))[0]

        tss_strt = [bar[m21.meter.TimeSignature] for bar in bars_1st]
        assert any(len(time_sigs) > 0 for time_sigs in tss_strt)

        self._vertical_bars: list[MxlMelodyExtractor.VerticalBar] = []

        self.tokenizer = MxlMelodyExtractor.Tokenizer(self.prec)

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
            for pnm, bar in obj.bars.items():
                if bar.hasVoices():
                    voices = bar.voices
                    ids = [v.id for v in voices]
                    # A bar may have just 2 voices
                    assert all(i in ids for i in ['1', '2'])
                    soprano = next(filter(lambda v: v.id == '1', voices))

                    # Insert notes in `soprano` by offset
                    obj.bars[pnm].remove(list(voices))
                    obj.bars[pnm].insert(flatten([e.offset, e] for e in soprano))
            for pnm, bar in obj.bars.items():
                assert not bar.hasVoices()
                notes = bar.notes
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
                assert not b[m21.note.Unpitched]
            pchs = self.avg_pitch(method=method, val_rest=val_rest)
            return max(self.bars, key=lambda p: pchs[p])

    def vertical_bars(self, scr):
        """
        :return: List of `VerticalBar`s
        """
        d_bars = {p.partName: list(sorted(p[m21.stream.Measure], key=lambda b: b.number)) for p in scr.parts}
        self.bar_strt_idx = list(d_bars.values())[0][0].number
        # All parts starts with the same bar number
        assert all(bars[0].number == self.bar_strt_idx for bars in d_bars.values())

        vbs = [self.VerticalBar({b.activeSite.partName: b for b in bars}) for bars in zip(*d_bars.values())]
        ts = vbs[0].time_sig  # Get Score starting time signature
        assert ts is not None
        for vb in vbs:  # Unroll time signature to each VerticalBar
            if vb.time_sig is None:
                vb.time_sig = ts
            else:
                ts = vb.time_sig
        return vbs

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
                    # ic(self.enc[obj.pitch.midi])
                    # exit(1)
                    return self.enc[obj.pitch.midi]
                elif isinstance(obj, m21.note.Rest):
                    return self.enc['[REST]']
                elif isinstance(obj, str) and obj in self.spec:
                    return self.enc[obj]
                else:
                    raise ValueError(f'Unexpected object: {obj}')

            def decode(self, obj) -> Union[int, np.integer, str]:
                # return self.spec_vocab_dec[i] if i in self.spec_vocab_dec else i - self.n_special
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
                for e in group_triplets(bar):
                    if isinstance(e, list):  # Triplet case
                        lst = e
                        dur = sum(e.duration.quarterLength for e in lst)  # Over `Fraction`s
                        assert dur.denominator == 1
                        dur = dur.numerator
                        # The smallest duration of triplet that can be encoded is 4 time slots
                        num_ea = (dur / 4 * n_slots_per_beat)
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
                        num = n_slots_per_beat*dur + 1
                        assert num.is_integer()
                        idxs_ = (np.linspace(strt, strt+dur, num=int(num)) * n_slots_per_beat)[:-1]
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
                ids: list[list[int]],
                time_sigs: list[m21.meter.TimeSignature] = None
        ) -> list[m21.stream.Measure]:
            """
            :param ids: A list of token ids
            :param time_sigs: Time Signatures of each bar
                If not given, the one from last encoding call is used
            :return: A human-readable representation of `ids`
            """
            if time_sigs is None:
                time_sigs = self.time_sigs

            ids = np.asarray(ids)
            enc_sep = self.tokenizer('[SEP]')
            idxs = np.where(ids == enc_sep)[0]
            lst_ids = np.split(ids, idxs)
            assert len(lst_ids) == len(time_sigs)
            # All lists except the 1st one starts with the bar SEP encoding
            lst_ids = [(l if idx == 0 else l[1:]) for idx, l in enumerate(lst_ids)]
            # Each element as 2-tuple of (id list, count list)
            lst_ids_n_cts = [tuple(zip(*compress(list(l)))) for l in lst_ids]
            return [
                self._decode(ids_n_cnt, time_sig, number=i)
                for i, (ids_n_cnt, time_sig) in enumerate(zip(lst_ids_n_cts, time_sigs))
            ]

        def _decode(
                self,
                ids_n_cnt: tuple[tuple[int], tuple[int]],
                time_sig: m21.meter.TimeSignature,
                number=None
        ):
            """
            For single bar
            """
            ic(number)
            kwargs = {} if number is None else dict(number=number)
            bar = m21.stream.Measure(**kwargs)
            n_slots_per_beat, n_slots = time_sig2n_slots(time_sig, self.prec)
            ids_, counts = ids_n_cnt
            durs = [count/n_slots_per_beat for count in counts]

            if number == 162:
                ic(ids_, durs)

            def id_n_dur2tok(i_, d_):
                return MxlMelodyExtractor.Tokenizer.Token(self.tokenizer.decode(i_), d_).note

            def get_toks():
                ids__ = np.asarray(ids_)
                # Indices for each triplet
                idxs_trip_end = np.where(ids__ == self.enc_trip)[0]
                idxs_trip_strt = idxs_trip_end - 3
                idxs_trip_strt[np.where(idxs_trip_strt < 0)[0]] = 0  # Starting indices should be at least 0
                if number == 162:
                    ic(idxs_trip_strt)
                if idxs_trip_strt.size >= 1:
                    l = ids__.size
                    idx = 0
                    lst_tok = []
                    while idx < l:
                        ic(idx)
                        if idx in idxs_trip_strt:
                            idx_end = idxs_trip_end[np_index(idxs_trip_strt, idx)]
                            dur_total = durs[idx_end] * 4  # The triplet encoding
                            assert dur_total.is_integer()
                            dur = Fraction(int(dur_total), 3)

                            ln = idx_end - idx
                            if ln == 3:  # Normal case
                                for id__ in ids__[idx:idx+3]:
                                    lst_tok.append(id_n_dur2tok(id__, dur))
                                idx += 4
                            else:  # Multiple contiguous triplet pitches with the same pitch
                                dur_ = sum(durs[idx:idx_end])
                                if dur_ == dur_total:
                                    ic()
                                    exit(1)
                                else:  # 1st triplet pitch same as prior normal pitch
                                    dur_n_trip = dur_-dur_total + dur_total/4
                                    ic(dur_n_trip)
                                    lst_tok.append(id_n_dur2tok(ids__[idx], dur_n_trip))  # The Non-triplet note
                                    dur_total -= dur_n_trip

                                    if ln == 1:
                                        ic(dur)
                                        for i in range(3):
                                            lst_tok.append(id_n_dur2tok(ids__[idx], dur))
                                    # if number == 162:
                                    #     ic(idx_end)
                                    #     ic(durs, idx)
                                    #     ic(durs[idx + 3])
                                    #     ic(dur_total)
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
        composer = PROJ_NM
        scr.metadata.composer = composer

        # Pick a `Part` to replace elements one by one, the 1st part selected as it contains all metadata
        idx_part = 0
        scr.remove(list(filter(lambda p: p is not scr.parts[idx_part], scr.parts)))
        assert len(scr.parts) == 1
        part = scr.parts[0]
        pnm = part.partName

        for i in range(1, len(self.scr.parts)):
            assert len(self.scr.parts[i][m21.tempo.MetronomeMark]) == 0

        for idx, bar in enumerate(part[m21.stream.Measure]):
            vb = self.vertical_bars(self.scr)[idx].single()
            pnm_ = vb.pnm_with_max_pitch(method='fqs')
            assert bar.number == idx + self.bar_strt_idx
            if pnm_ != pnm:
                assert part.index(bar) == idx+1
                part.replace(bar, vb[pnm_])

        # Set instrument as Piano
        instr = m21.instrument.Piano()
        [part.remove(ins) for ins in part[m21.instrument.Instrument]]
        part.insert(instr)
        part.partName = f'{PROJ_NM}, {instr.instrumentName}, CH #1'

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

    def encoding2score(self, ids):
        bars = self.tokenizer.decode(ids)
        # Set default tempo
        bars[0].insert(m21.tempo.MetronomeMark(number=config('Melody-Extraction.output.BPM')))
        part = m21.stream.Part()
        instr = m21.instrument.Piano()
        part.partName = f'{PROJ_NM}, {instr.instrumentName}, CH #1'
        part.append(instr)
        part.append(bars)

        scr = m21.stream.Score()
        scr.append(m21.metadata.Metadata())
        scr.metadata.composer = PROJ_NM
        scr.metadata.title = f'{self.score_title}, decoded'
        scr.append(part)
        scr.show()


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
        fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        # fnm = eg_songs('Shape of You', fmt='MXL')
        ic(fnm)
        me = MxlMelodyExtractor(fnm, n=None)
        me.bar_with_max_pitch(exp='symbol')
    # extract_encoding()

    def sanity_check_encoding():
        fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        # fnm = eg_songs('Shape of You', fmt='MXL')
        ic(fnm)
        me = MxlMelodyExtractor(fnm, n=None)
        ids = me.bar_with_max_pitch(exp='symbol')
        ic(ids[:10])
        ic(me.encoding2score(ids))
    sanity_check_encoding()

