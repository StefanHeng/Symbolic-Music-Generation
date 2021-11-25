import math
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
        # ic(dir(notes[0].duration))
        # ic(notes[0].duration)
        # ic(notes[0].duration.quarterLength)
        end = notes[0].offset + notes[0].duration.quarterLength
        for note in notes[1:]:
            # Current note should begin, not before the previous one ends
            # if end > note.offset:
            #     ic('large', end, note.offset, end > note.offset, math.isclose(end, note.offset, abs_tol=1e-6))
            #     for n in notes:
            #         res = n.offset + n.duration.quarterLength
            #         ic(n, n.duration, type(n.duration), n.duration.quarterLength)
            #         ic(n.offset, end)
            #     exit(1)
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

        def single(self):  # TODO: `, inplace=False` ?
            """
            For each time step in each bar, filter notes such that only the note with highest pitch remains

            Chords are effectively converted to notes

            .. note:: The `bars` attribute is modified
            """
            # ic(self.bars[list(self.bars.keys())[0]].number)
            for pnm, bar in self.bars.items():
                # bar.show(fmt='musicxml.png')
                notes = bar.notes
                assert notes.isSorted
                # notes = sorted(bar.notes, key=lambda n: n.offset)
                # ic(notes)

                if notes:
                    # chd = notes[0]
                    # ic(chd, chd.pitches, chd.notes)
                    # ic(chd.notes[0].pitch.midi, chd.notes[1].pitch.midi)
                    # nt = chd.notes[0]
                    # ic(nt, nt.pitch, nt.pitch.midi, nt.pitch.frequency)
                    assert_notes_no_overlap(notes)
                    # for n in notes:
                    #     ic(n.isNote)

                def chord2note(c):
                    return max(c.notes, key=lambda n: n.pitch.frequency)
                # m21.stream.Measure
                # for elm in bar:
                #     ic(elm)
                # notes_ = list(map(lambda elm: chord2note(elm) if isinstance(elm, m21.chord.Chord) else elm, notes))
                # # self.bars[pnm].remove(notes)
                #
                # if not list(bar):
                #     ic(list(bar))
                # ic(list(bar))
                # bar.remove(list(notes))
                # ic(list(bar), notes_)
                if notes:
                    ic(list(bar))
                    # bar.show()
                    for note in notes:
                        ic([p.octave for p in note.pitches])
                        # ic(note.pitch.octave)
                        if isinstance(note, m21.chord.Chord):
                            bar.replace(note, chord2note(note))
                    ic(list(bar))
                    for note in bar.notes:
                        ic(note.pitch.octave)

                    exit(1)

    def pitch_avg(self):
        """
        :return: Average pitch, weighted by duration, for each bar

        Pitch value by either MIDI representation integer, or frequency
        """

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
            ts = vbs[0].time_sig
            assert ts is not None
            for vb in vbs:  # Unroll time signature to each VerticalBar
                if vb.time_sig is None:
                    vb.time_sig = ts
                else:
                    ts = vb.time_sig
            # ic([vb.time_sig for vb in vbs])
            self._vertical_bars = vbs
        return self._vertical_bars

    def bar_with_max_pitch(self):
        """
        For each bar, pick the track with highest average pitch
        """
        for vb in self.vertical_bars:
            # if vb.n == 83:
            #     list(vb.bars.values())[0].show()
            vb.single()

        s = m21.stream.Score()
        # Per `music21`, duration is represented in terms of quarter notes, definitely an integer
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

