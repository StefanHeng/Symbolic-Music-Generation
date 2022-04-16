from enum import Enum
from typing import Dict
from collections import namedtuple


class ElmType(Enum):
    bar_start, song_end, time_sig, tempo, note, tuplets = list(range(6))


class Key(Enum):
    """
    `f` for flat, `s` for sharp

    See musicnlp.preprocess.key_finder.py
    """
    CMaj, FMaj, BfMaj, EfMaj, AfMaj, DfMaj, GfMaj, BMaj, EMaj, AMaj, DMaj, GMaj, \
        AMin, DMin, GMin, CMin, FMin, BfMin, EfMin, GsMin, CsMin, FsMin, BMin, EMin = list(range(24))

    @classmethod
    def from_str(cls, key: str) -> 'Key':
        return key2ordinal[key]


key2ordinal: Dict[str, Key] = {
    'CMajor': Key.CMaj,
    'FMajor': Key.FMaj,
    'BbMajor': Key.BfMaj,
    'EbMajor': Key.EfMaj,
    'AbMajor': Key.AfMaj,
    'DbMajor': Key.DfMaj,
    'GbMajor': Key.GfMaj,
    'BMajor': Key.BMaj,
    'EMajor': Key.EMaj,
    'AMajor': Key.AMaj,
    'DMajor': Key.DMaj,
    'GMajor': Key.GMaj,
    'AMinor': Key.AMin,
    'DMinor': Key.DMin,
    'GMinor': Key.GMin,
    'CMinor': Key.CMin,
    'FMinor': Key.FMin,
    'BbMinor': Key.BfMin,
    'EbMinor': Key.EfMin,
    'G#Minor': Key.GsMin,
    'C#Minor': Key.CsMin,
    'F#Minor': Key.FsMin,
    'BMinor': Key.BMin,
    'EMinor': Key.EMin,
}


# an intermediate representation, for conversion between music string & MXL
MusicElement = namedtuple(typename='MusicElement', field_names=['type', 'meta'], defaults=[None, None])
