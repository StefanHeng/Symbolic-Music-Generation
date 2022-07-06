from enum import Enum
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass


__all__ = [
    'ElmType', 'MusicElement',
    'Key', 'key_str2enum', 'key_enum2tuple', 'key_str2ordinal', 'key_ordinal2str', 'key_ordinal2key_enum',
    'key_offset_dict', 'OFFKEY_OFFSET'
]


class ElmType(Enum):
    bar_start, song_end, time_sig, tempo, key, note, tuplets = list(range(7))


@dataclass
class MusicElement:
    """
    an intermediate representation, for conversion between music string & MXL
    """
    type: ElmType
    meta: Optional[Union[int, Tuple]] = None


class Key(Enum):
    """
    `f` for flat, `s` for sharp

    See musicnlp.preprocess.key_finder.py
    """
    CMaj, FMaj, BfMaj, EfMaj, AfMaj, DfMaj, GfMaj, BMaj, EMaj, AMaj, DMaj, GMaj, \
        AMin, DMin, GMin, CMin, FMin, BfMin, EfMin, GsMin, CsMin, FsMin, BMin, EMin = list(
            range(24))

    @classmethod
    def from_str(cls, key: str) -> 'Key':
        return key_str2enum[key]


key_str2enum: Dict[str, Key] = {
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

# Dictionary for translating a Key type to a tuple of ints (type, key)
# where type := major (1) or minor (0) and key := the name of the keyC
key_enum2tuple: Dict[Key, Tuple[int, str]] = {
    Key.CMin: (0, 'C'),
    Key.CsMin: (0, 'C#'),
    Key.DMin: (0, 'D'),
    Key.EfMin: (0, 'E-'),
    Key.EMin: (0, 'E-'),
    Key.FMin: (0, 'F'),
    Key.FsMin: (0, 'F#'),
    Key.GMin: (0, 'G'),
    Key.GsMin: (0, 'G#'),
    Key.AMin: (0, 'A'),
    Key.BfMin: (0, 'B-'),
    Key.BMin: (0, 'B'),
    Key.CMaj: (1, 'C'),
    Key.DMaj: (1, 'D'),
    Key.DfMaj: (1, 'D-'),
    Key.EfMaj: (1, 'E-'),
    Key.EMaj: (1, 'E'),
    Key.FMaj: (1, 'F'),
    Key.GMaj: (1, 'G'),
    Key.GfMaj: (1, 'G-'),
    Key.AMaj: (1, 'A'),
    Key.AfMaj: (1, 'A-'),
    Key.BfMaj: (1, 'B-'),
    Key.BMaj: (1, 'B')
}

# for passing key during vanilla training to monitor IKR
key_str2ordinal: Dict[str, int] = {k: i for i, k in enumerate(key_str2enum.keys())}
key_ordinal2str: Dict[int, str] = {i: k for k, i in key_str2ordinal.items()}
key_ordinal2key_enum: Dict[int, Key] = {i: key_str2enum[k] for k, i in key_str2ordinal.items()}

# This does not take ENHARMONICS into account
# TODO: Fix this by adding COMPLETE ENHARMONIC relations
key_offset_dict: Dict[str, int] = {
    'C': 0,
    'C#': 1,
    'D-': 1,
    'D': 2,
    'D#': 3,
    'E-': 3,
    'E': 4,
    'F': 5,
    'F#': 6,
    'G-': 6,
    'G': 7,
    'G#': 8,
    'A-': 8,
    'A': 9,
    'B-': 10,
    'B': 11
}
MAJOR_OFFKEY_OFFSET_IDX = [1, 3, 6, 8, 10]
# Harmonic key (INKEY) offset for MINOR: [0, 2, 3, 5, 7, 8, (10 or 11)]
MINOR_OFFKEY_OFFSET_IDX = [1, 4, 6, 9, 11]
OFFKEY_OFFSET = [MINOR_OFFKEY_OFFSET_IDX, MAJOR_OFFKEY_OFFSET_IDX]


if __name__ == '__main__':
    assert len(key_str2enum) == len(
        key_enum2tuple), "Dictionary should be of the same size"
