from icecream import ic
from util import *


if __name__ == '__main__':
    import os

    os.chdir('../../datasets')
    fnm = config('datasets.Allie_Chords')
    ic(fnm)

    p = read_pickle('full_song_objects.pickle')[0]
    ic(len(p))

