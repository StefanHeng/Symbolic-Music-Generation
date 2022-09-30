from musicnlp.util.util import *


if __name__ == '__main__':
    import os

    from stefutil import mic

    os.chdir('../../../datasets')
    d = config('datasets.Allie_Chords')
    fnm = f'{d["dir_nm"]}/{d["nm_data"]}'
    mic(fnm)

    p = read_pickle(fnm)[0]
    mic(len(p))

