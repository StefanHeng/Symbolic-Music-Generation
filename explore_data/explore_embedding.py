import os
import pickle
from icecream import ic


def read_pickle(fnm):
    objects = []
    with (open(fnm, 'rb')) as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    return objects


if __name__ == '__main__':
    os.chdir('../..')

    p = read_pickle('datasets/full_song_objects.pickle')[0]
    ic(len(p))

