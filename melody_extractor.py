from mido import MidiFile
import pretty_midi
from pretty_midi import PrettyMIDI

from util import *


class MelodyExtractor:
    """
    Given MIDI file, export single-track melody representations, as matrix or MIDI file

    Only MIDI files with no tempo change are considered

    Velocity of all notes assumed the same

    Enforce that at each time step, there will be only one note played

    The pitch of each note follows the MIDI standard, as integer in [0, 127]
    """
    def __init__(self, fl_nm, precision=2**5):
        """
        :param fl_nm: Path & MIDI file name
        :param precision: 1/`precision` would be the smallest unit per bar
        """
        self.fnm = fl_nm
        self.precision = precision
        self.mu = MidoUtil()
        self.pmu = PrettyMidiUtil()
        self.bpm = self._get_bpm()

    def _get_bpm(self):
        if not hasattr(self, 'mido'):
            self.mido = MidiFile(self.fnm)

        tempos = self.mu.get_tempo_changes(self.mido)
        ic(tempos)
        assert len(tempos) == 1
        return tempo2bpm(tempos[0])

    def bar_with_max_pitch(self):
        """
        For each bar, pick the track with highest average pitch

        If multiple notes at a time step,
        """


if __name__ == '__main__':
    from icecream import ic

    fnm = eg_midis(2)
    ic(fnm)
    me = MelodyExtractor(fnm)
    ic(me.bpm)
