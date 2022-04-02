import os

DIR_DSET = 'datasets'  # Dataset root folder name
DIR_MDL = 'models'

paths = __file__.split(os.sep)
paths = paths[:paths.index('util')]

# Absolute system path for root directory;
#   e.g.: '/Users/stefanh/Documents/UMich/Research/Music with NLP'
PATH_BASE = os.sep.join(paths[:-2])  # System data path
# Repo root folder name with package name; e.g.: 'Symbolic-Music-Generation/musicnlp'
DIR_PROJ = paths[-2]
PKG_NM = paths[-1]  # Package/Module name


if __name__ == '__main__':
    from icecream import ic
    ic(PATH_BASE, DIR_PROJ, DIR_DSET, DIR_MDL, PKG_NM)
