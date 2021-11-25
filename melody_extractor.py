import math
from copy import deepcopy
from warnings import warn
from typing import Union

from mido import MidiFile
import pretty_midi
from pretty_midi import PrettyMIDI
import music21 as m21

from util import *


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
        For each bar, pick the track with highest average pitch

        If multiple notes at a time step, pick the one with highest pitch
        """
        # ic(self.precision - self.frac_per_beat, (self.precision - self.frac_per_beat)**2)
        # ic(spb, spt, 1/spt)
        # pr = self.pm.get_piano_roll(fs=)
        # ic(self.pm.instruments[0].notes[:10])
        pr = self.pm.get_piano_roll(fs=self.fqs_ts)
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
    the concurrent notes are filtered such that only the note with highest pitch remains

    The pitch will be encoded as integer in [0-127] per MIDI convention
    """
    def __init__(self, fl_nm, precision=5):
        self.fnm = fl_nm
        self.prec = precision

        self.scr = m21.converter.parse(self.fnm)
        lens = [len(p[m21.stream.Measure]) for p in self.scr.parts]
        assert_list_same_elms(lens)

        self._vertical_bars: list[MxlMelodyExtractor.VerticalBar] = []

    class VerticalBar:
        """
        Contains all the bars for each channel in the score, at the same time
        """
        def __init__(self, bars: dict[str, m21.stream.Measure]):
            self.bars = bars
            nums = [bar.number == 0 for bar in bars.values()]
            assert_list_same_elms(nums)
            self.n = nums[0]

            tss = [b.timeSignature for b in self.bars.values()]
            self._time_sig = None
            if tss[0] is not None:
                dss = [(ds.numerator, ds.denominator) for ds in tss]
                assert_list_same_elms(dss)
                self._time_sig = dss[0]

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
            For each time step in each bar, filter notes such that only the note with highest pitch remains

            Chords are effectively converted to notes

            .. note:: The `bars` attribute is modified
            """
            # obj = None if inplace else deepcopy(self)
            obj = self if inplace else deepcopy(self)
            for pnm, bar in obj.bars.items():
                if bar.hasVoices():
                    voices = bar.voices
                    ids = [v.id for v in voices]
                    assert all(i in ids for i in ['1', '2', '3', '4'])
                    soprano = next(filter(lambda v: v.id == '1', voices))

                    # Get the correct index to insert notes for `soprano`
                    min_voice = min(voices, key=lambda v: obj.bars[pnm].index(v))
                    idx = obj.bars[pnm].index(min_voice)
                    obj.bars[pnm].remove(list(voices))
                    obj.bars[pnm].insert(alternate(range(idx, idx+len(soprano)), list(soprano)))
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

            ic(self.bars)

            def _avg_pitch(b):
                # ic('here', b)
                if b.notes:
                    fs = [f(n) for n in b.notes]
                    ws = [n.duration.quarterLength for n in b.notes]
                    fs.append(val_rest)
                    ws.append(sum([r.quarterLength for r in b[m21.note.Rest]]))
                    ic(fs, ws)
                    # for n in b:
                    #     ic(n, n.quarterLength)
                    return np.average(np.array(fs), weights=np.array(ws))
                else:
                    return float('-inf')
            # _avg_pitch(list(self.bars.values())[2])
            # _avg_pitch(self.bars['Piano, CH #2'])
            #
            # ic('here')
            return {pnm: _avg_pitch(bar) for pnm, bar in self.items()}
            # return {
            #     pnm: np.average(
            #         np.array([f(n) for n in bar.notes]), weights=np.array([n.duration.quarterLength for n in bar.notes])
            #     ) if len(bar.notes) > 0 else float('-inf')
            #     for pnm, bar in self.bars.items()
            # }

        def pnm_with_max_pitch(self, method='fqs', val_rest=0):
            """
            :return: The part name key, that has the maximum pitch, per `avg_pitch`
            """
            pchs = self.avg_pitch(method=method, val_rest=val_rest)
            ic(pchs)
            return max(self.bars, key=lambda p: pchs[p])

        def __getitem__(self, key):
            return self.bars[key]

        def __iter__(self):
            return iter(self.bars.keys())

        def values(self):
            return self.bars.values()

        def items(self):
            return self.bars.items()

    @property
    def vertical_bars(self):
        """
        :return: List of `VerticalBar`s
        """
        if not self._vertical_bars:
            pnms = [p.partName for p in self.scr.parts]
            assert len(pnms) == len(set(pnms))  # Unique part names
            d_bars = {p.partName: list(sorted(p[m21.stream.Measure], key=lambda b: b.number)) for p in self.scr.parts}
            assert all(bars[0].number == 0 for bars in d_bars.values())  # All parts starts bar number with 0

            vbs = [self.VerticalBar({b.activeSite.partName: b for b in bars}) for bars in zip(*d_bars.values())]
            ts = vbs[0].time_sig  # Get Score starting time signature
            assert ts is not None
            for vb in vbs:  # Unroll time signature to each VerticalBar
                if vb.time_sig is None:
                    vb.time_sig = ts
                else:
                    ts = vb.time_sig
            self._vertical_bars = vbs
        return self._vertical_bars

    def bar_with_max_pitch(self, exp='stream'):
        """
        :param exp: Export format,
            either `stream` for `music21.stream.Score` or `symbol` for symbolic encodings
        For each bar, pick the track with highest average pitch
        """
        scr = deepcopy(self.scr)
        scr.metadata.composer = PROJ_NM
        # Pick one `Part` arbitrarily to replace elements one by one
        idx_part_map = 0
        scr.remove(list(filter(lambda p: p is not scr.parts[idx_part_map], scr.parts)))
        assert len(scr.parts) == 1
        part = scr.parts[0]
        pnm = part.partName
        ic(pnm)
        # ic(list(self.scr.parts[0][m21.tempo.MetronomeMark]))
        # ic(list(self.scr.parts[1][m21.tempo.MetronomeMark]))
        vb = self.vertical_bars[6].single()
        ic(vb.pnm_with_max_pitch(method='midi'))
        exit(1)
        for pnm, bar in self.vertical_bars[6].items():
            ic(pnm, list(bar))
            # ic(list(bar.voices))
            if bar.voices:
                ic(bar.isSorted)
                # bar.show(fmt='musicxml.png')
                for v in bar.voices:
                    ic(type(v))
                    v: m21.stream.Voice
                    # ic(v, vars(v))
                    # ic(list(v.notes))
                    ic(v)
                    for note in v.notes:
                        ic(note, note.offset, note.duration)
        exit(1)
        for idx, bar in enumerate(part[m21.stream.Measure]):
            if idx == 6:
                vb = self.vertical_bars[idx]
                for pnm, b in vb.items():
                    ic(pnm, b)
                    if '2' in pnm:
                        # b.show(fmt='musicxml.png')
                        ic(list(b))
                        ic(b.flatten())
                        ic(list(b.recurse()))
                exit(1)
            vb = self.vertical_bars[idx].single()
            pnm_ = vb.pnm_with_max_pitch(method='midi')
            assert bar.number == idx
            if idx == 6:
                bar.show(fmt='musicxml.png')
                vb[pnm_].show(fmt='musicxml.png')
            if pnm_ != pnm:
                assert part.index(bar) == idx+1  # Since the first element in a `Part` is the
                part.replace(bar, vb[pnm_])

        part.partName = f'{PROJ_NM}, CH #1'
        part.show()

        if exp == 'symbol':
            # Per `music21`, duration is represented in terms of quarter notes
            slot_dur = int(2**-2 / 2**-self.prec)  # Duration of a time slot
            ic(slot_dur)

    def slot_with_max_pitch(self):
        """
        For each time slot, pick track with highest pitch
        """
        pass


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
        fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        ic(fnm)
        me = MxlMelodyExtractor(fnm)
        me.bar_with_max_pitch()
    check_mxl()

