from musicnlp.util.util import *


if __name__ == '__main__':
    import os

    os.chdir('../../../datasets')
    d = config('datasets.Allie_Chords')
    fnm = f'{d["dir_nm"]}/{d["nm_data"]}'
    ic(fnm)

    p = read_pickle(fnm)[0]
    ic(len(p))

