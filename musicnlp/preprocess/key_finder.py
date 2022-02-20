"""Given a piece in musicxml format, return the key of the piece
 using the Krumhansl-Schmuckler key-finding algorithm.
 TODO:What if there are some other good algos?
 It still would not realize all 24 keys!(only 12)"""
import music21 as m21
import numpy as np
from icecream import ic


def get_durations(s):
    """
    s: a music21.Stream object that stores the piece without drums
    return: a np list of total durations for each pitch class in quarterLength.
    P.S. So kind of normalized version?
    """
    # flatten, then filter all the notes
    result = np.zeros(12)
    for n in s.flatten().flatten().notesAndRests:
        length = n.quarterLength
        if n.isChord:
            for m in n.pitchClasses:
                result[m] += length
        elif not n.isRest:
            result[n.pitch.pitchClass] += length
    ic(result)
    return result


class KeyFinder:
    """
    Given a MusicXML file, find the key of those pieces.
    TODO: Do I need to find all modulated keys?
    """

    def __init__(self, file_name):
        """file_name: the name of file given path it is in"""
        self.piece = m21.converter.parse(file_name)

        # remove all the percussion in this piece, got from MelodyExtractor.py
        def is_drum(part):
            """
            :return: True if `part` contains *only* `Unpitched`
            """
            return list(part[m21.note.Unpitched]) and not list(part[m21.note.Note])

        parts_drum = filter(lambda p_: any(p_[drum] for drum in [
            m21.instrument.BassDrum,
            m21.instrument.BongoDrums,
            m21.instrument.CongaDrum,
            m21.instrument.SnareDrum,
            m21.instrument.SteelDrum,
            m21.instrument.TenorDrum,
        ]) or is_drum(p_), self.piece.parts)
        for pd in parts_drum:
            self.piece.remove(pd)

        # major and minor profile, see http://rnhart.net/articles/key-finding/
        self.prof = np.array([[0.748, 0.06, 0.488, 0.082, 0.67, 0.46, 0.096, 0.715, 0.104, 0.366, 0.057, 0.4],
                              [0.712, 0.084, 0.474, 0.618, 0.049, 0.46, 0.105, 0.747, 0.404, 0.067, 0.133, 0.33]])

    # @eye
    def find(self):
        """
        return: key of the piece as a string.
        """
        tonality = ['Major', 'Minor']
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        durations = get_durations(self.piece)
        # initialize results for all 24 possible coefficients
        corrcoef_mat = np.empty((2, 12))
        for k in range(2):
            for i in range(12):
                # linear correlation: https://realpython.com/numpy-scipy-pandas-correlation-python/#linear-correlation
                # also remember to rotate the weight matrix couple times
                corrcoef_mat[k, i] = np.corrcoef(np.roll(self.prof[k], i), durations)[1][0]
        ic(corrcoef_mat)
        best_val_maj = np.max(corrcoef_mat[0])
        best_val_min = np.max(corrcoef_mat[1])
        # fuzzy search
        close_ma = len(corrcoef_mat[0][corrcoef_mat[0] >= best_val_maj * 0.8])
        close_mi = len(corrcoef_mat[1][corrcoef_mat[1] >= best_val_min * 0.8])
        best_maj_keys = (np.argsort(corrcoef_mat[0]))[-close_ma:]
        best_min_keys = (np.argsort(corrcoef_mat[1]))[-close_mi:]
        maj_keys_result = [f'{pitches[tonic]}Major' for (_, tonic) in
                           [divmod(i, 12) for i in best_maj_keys]]
        min_keys_result = [f'{pitches[tonic]}Minor' for (_, tonic) in
                           [divmod(i, 12) for i in best_min_keys]]
        return maj_keys_result, min_keys_result
    #
    # def alt_find(self):
    #     a = m21.analysis.discrete.TemperleyKostkaPayne(self.piece)
    #     print(a.getSolution(self.piece))
    #     return m21.analysis.discrete.TemperleyKostkaPayne(self.piece)


def main():
    a = KeyFinder('/Users/carsonzhang/Desktop/Projects/Rada/midi/Merry-Go-Round-of-Life.musicxml')
    ic(a.find())


if __name__ == '__main__':
    main()
