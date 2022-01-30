"""
Since Sun. Jan. 30th, an updated module for music/melody extraction, with a duration-quantized approach

See `melody_extractor` for the old version.
"""


from musicnlp.util import *


class MusicExtractor:
    """
    Extract melody and potentially chords from MXL music scores => An 1D polyphonic representation
    """
    def __init__(self, scr: Union[str, m21.stream.Score], precision: int = 5):
        """
        :param scr: A music21 Score object, or file path to an MXL file
        :param precision: Bar duration quantization, see `melody_extractor.MxlMelodyExtractor`
        """
        if isinstance(scr, str):
            self.scr = m21.converter.parse(scr)
        else:
            self.scr = scr
        self.scr: m21.stream.Score

        self.prec = precision

    def __call__(self, exp='mxl'):
        pass


if __name__ == '__main__':
    from icecream import ic

    def toy_example():
        # fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        fnm = eg_songs('Shape of You', fmt='MXL')
        ic(fnm)
        me = MusicExtractor(fnm)
        me(exp='mxl')
    toy_example()
