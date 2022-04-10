from enum import Enum
from collections import namedtuple


class ElmType(Enum):
    bar_start, song_end, time_sig, tempo, note, tuplets = list(range(6))


# an intermediate representation, for conversion between music string & MXL
MusicElement = namedtuple(typename='MusicElement', field_names=['type', 'meta'], defaults=[None, None])
