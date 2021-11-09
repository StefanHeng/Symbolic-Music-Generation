import json
import pickle

from data_path import *


def read_pickle(fnm):
    objects = []
    with (open(fnm, 'rb')) as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    return objects


def config(attr):
    """
    Retrieves the queried attribute value from the config file.

    Loads the config file on first call.
    """
    if not hasattr(config, 'config'):
        with open(f'{PATH_BASE}/config.json') as f:
            config.config = json.load(f)

    node = config.config
    for part in attr.split('.'):
        node = node[part]
    return node
