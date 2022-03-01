from enum import Enum


class ElmType(Enum):
    bar_start, song_end, time_sig, tempo, note, tuplets = list(range(6))
