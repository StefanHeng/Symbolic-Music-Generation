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
    # ic(result)
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

        # diatonic key naming convention, see 'Circle of Fifth'.
        self.conv_major = {
            'C': 'C',
            'F': 'F',
            'A#': 'Bb',
            'D#': 'Eb',
            'G#': 'Ab',
            'C#': 'Db',
            'F#': 'Gb',
            'B': 'B',
            'E': 'E',
            'A': 'A',
            'D': 'D',
            'G': 'G'
        }
        self.conv_minor = {
            'A': 'A',
            'D': 'D',
            'G': 'G',
            'C': 'C',
            'F': 'F',
            'A#': 'Bb',
            'D#': 'Eb',
            'G#': 'G#',
            'C#': 'C#',
            'F#': 'F#',
            'B': 'B',
            'E': 'E'
        }

    # @eye
    def find_key(self):
        """
        return: 2 arrays that contains the best k candidates for major and minor respectively
        of the piece as a string.
        The string format would be [keyName]+Major/Minor.
        All keys with accidental signs are marked as sharp, which would equate 'A#' to 'Bb'.
        Then be transformed to more conventional enharmonic reading. e.g. 'A#' to 'Bb'..
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
        # ic(corrcoef_mat)
        best_val_maj = np.max(corrcoef_mat[0])
        best_val_min = np.max(corrcoef_mat[1])
        # fuzzy search
        close_ma = len(corrcoef_mat[0][corrcoef_mat[0] >= best_val_maj * 0.8])
        close_mi = len(corrcoef_mat[1][corrcoef_mat[1] >= best_val_min * 0.8])
        best_maj_keys = (np.argsort(corrcoef_mat[0]))[-close_ma:]
        best_min_keys = (np.argsort(corrcoef_mat[1]))[-close_mi:]
        # convert candidates to string in convention format(circle of fifth).
        maj_keys_result = [f'{self.conv_major[pitches[tonic]]}Major' for (_, tonic) in
                           [divmod(i, 12) for i in best_maj_keys]]
        min_keys_result = [f'{self.conv_minor[pitches[tonic]]}Minor' for (_, tonic) in
                           [divmod(i, 12) for i in best_min_keys]]
        #
        return maj_keys_result, min_keys_result
    #
    # def alt_find(self):
    #     a = m21.analysis.discrete.TemperleyKostkaPayne(self.piece)
    #     print(a.getSolution(self.piece))
    #     return m21.analysis.discrete.TemperleyKostkaPayne(self.piece)

    def find_scale_degrees(self, k):
        """
        k: tuple of 2 lists, each contains major keys candidates and minor keys candidates ([XMajor],[YMinor])
        Output: a dictionary of s in scale degrees of given key in k represented in tuple where each tuple has
        (note name/pitch, scale degrees) **note: they do not have any octave values!
        """
        # make a dictionary with group T0 in scale degrees
        # Set e in T0 group, in this case it will be C
        # **note: the notion of transposition group has been abused here and forced to adapt to enharmonic scale
        T_0 = {
            'C': 0,
            'D': 1,
            'E': 2,
            'F': 3,
            'G': 4,
            'A': 5,
            'B': 6,
        }
        T_1 = {
            'C': 1,
            'D': 2,
            'E': 3,
            'F': 4,
            'G': 5,
            'A': 6,
            'B': 7,
        }
        piece = self.piece
        all_k = k[0] + k[1]
        # to store all scale degrees in T0
        arr_ = []
        for n in piece.flatten().flatten().notesAndRests:
            if n.isChord:
                for p in n.pitches:
                    arr_.append((p.name, T_0[p.step]))
            elif n.isRest:
                arr_.append(('R', 0))
            else:
                arr_.append((n.name, T_0[n.step]))
        # now shift to T1 and adjust the scale degree accordingly with major and minor
        ret_ = {}
        for k_ in all_k:
            step = k_[0]
            ret_[k_] = [(name, scale+T_1[step]) if name != 'R' else (name, scale) for name, scale in arr_]
        return ret_

    def check_notes(self, k):
        """
        There are 3 kinds of common dissonance in classical period:
        1. modal mixture
        2. Secondary dominant
        3. Neapolitan chord in minor
        Here we will only consider the first two case since the 3rd has been deprecated by modern music.
        TODO: It is very tricky to do such analysis, need to talk with group.
        """
        pass


def main(path: str):
    a = KeyFinder(path)
    k = a.find_key()
    ic(k)
    ic(a.find_scale_degrees(k))


if __name__ == '__main__':
    from icecream import ic

    import musicnlp.util.music as music_util

    p = music_util.get_my_example_songs('Merry Go Round of Life', fmt='MXL')
    ic(p)

    main(p)
