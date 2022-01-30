import os

DIR_DSET = 'datasets'  # Dataset root folder name

paths = __file__.split(os.sep)
paths = paths[:paths.index('util')]

# Absolute system path for root directory;
#   e.g.: '/Users/stefanh/Documents/UMich/Research/Music with NLP'
PATH_BASE = os.sep.join(paths[:-2])  # System data path
# Repo root folder name with package name; e.g.: 'Symbolic-Music-Generation/musicnlp'
DIR_PROJ = os.sep.join(paths[-2:])
