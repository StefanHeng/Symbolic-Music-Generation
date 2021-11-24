from math import log2

from mido import MidiFile
import pretty_midi
from pretty_midi import PrettyMIDI
import music21 as m21

from util import *


def assert_list_same_elms(lst):
    assert all(l == lst[0] for l in lst)


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
        assert log2(self.frac_per_beat).is_integer()
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

    Each bar is divided into equidistant slots of music note length, given by a `precision` attribute
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
        # lens = [len(p.measures(numberStart=0, numberEnd=None, collect=[])) for p in self.scr.parts]
        lens = [len(p[m21.stream.Measure]) for p in self.scr.parts]
        # ic(lens)
        assert_list_same_elms(lens)
        # assert all(l == lens[0] for l in lens)  # Same number of bar for each part
        # for p in self.scr.parts:
        #     bars = p.measures(numberStart=0, numberEnd=None, collect=[])
        #     for bar in bars:
        #         ic(bar)

    class VerticalBar:
        """
        Contains all the bars for each channel in the score, at the same time
        """
        def __init__(self, bars: dict[str, m21.stream.Measure]):
            self.bars = bars
            # ic(bars)
            tss = [b.timeSignature for b in self.bars.values()]
            # .displaySequence
            # tss = [(ds.numerator, ds.denominator) for ds in dss]
            # ic(tss)
            # # time_sig =
            # ts = tss[0]
            # ic(vars(ts))
            # ms: m21.meter.core.MeterSequence = ts.displaySequence
            # ic()
            # ic(vars(ts.displaySequence))
            # assert_list_same_elms(tss)
            self._time_sig = None
            if tss[0] is not None:
                dss = [(ds.numerator, ds.denominator) for ds in tss]
                assert_list_same_elms(dss)
                # ic(dss)
                self._time_sig = dss[0]

        @property
        def time_sig(self):
            return self._time_sig

        @time_sig.setter
        def time_sig(self, val):
            self._time_sig = val

    @property
    def vertical_bars(self):
        """
        :return: A list of scores
        """
        # d_bars =
        # ic(len(self.scr))
        pnms = [p.partName for p in self.scr.parts]
        assert len(pnms) == len(set(pnms))  # Unique part names
        d_bars = {p.partName: list(sorted(p[m21.stream.Measure], key=lambda b: b.number)) for p in self.scr.parts}
        assert all(bars[0].number == 0 for bars in d_bars.values())  # All parts starts bar number with 0

        # for k, v in d_bars.items():
        #     bar = v[0]
        #     # ic(vars(bar))
        #     ic(bar.activeSite.partName)
        #     exit(1)
        # ic(d_bars)
        # for part in self.scr.parts:
        #     ic(part)
        #     ic(part.partName)
        #     bars = part[m21.stream.Measure]
        #     ic(len(bars))
        #     for bar in bars:
        #         ic(bar.number)
        # for i in zip(d_bars.keys(), zip(*d_bars.values())):
        #     ic(i)
        # for bars in zip(*d_bars.values()):
        #     for b in bars:
        #         ic(b, b.activeSite)
        #     # ic({b.actieSite.partName: b for b in bars})
        #     ic([b.activeSite.partName for b in bars])
        #     exit(1)
        vbs = [self.VerticalBar({b.activeSite.partName: b for b in bars}) for bars in zip(*d_bars.values())]
        # bar = bars_by_time[0]['Piano, CH #2']
        # ic(bar, bar.timeSignature)
        ts = vbs[0].time_sig
        assert ts is not None
        for vb in vbs:  # Unroll time signature for each VerticalBar
            if vb.time_sig is None:
                vb.time_sig = ts
            else:
                ts = vb.time_sig
        ic([vb.time_sig for vb in vbs])
        return 'blah'

    def bar_with_max_pitch(self):
        """
        For each bar, pick the track with highest average pitch
        """
        bars = self.vertical_bars

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

