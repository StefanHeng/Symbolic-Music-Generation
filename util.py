import json
import pickle
import glob
from functools import reduce

import pandas as pd
from mido import MidiFile

from data_path import *


def flatten(lsts):
    """ Flatten list of list of elements to list of elements """
    return [e for lst in lsts for e in lst]


def read_pickle(fnm):
    objects = []
    with (open(fnm, 'rb')) as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    return objects


def get(dic, ks):
    """
    :param dic: Potentially multi-level dictionary
    :param ks: Potentially `.`-separated keys
    """
    ks = ks.split('.')
    return reduce(lambda acc, elm: acc[elm], ks, dic)


def set_(dic, ks, val):
    ks = ks.split('.')
    node = reduce(lambda acc, elm: acc[elm], ks[:-1], dic)
    node[ks[-1]] = val


def keys(dic, prefix=''):
    """
    :return: Generator for all potentially-nested keys
    """
    def _full(k_):
        return k_ if prefix == '' else f'{prefix}.{k_}'
    for k, v in dic.items():
        if isinstance(v, dict):
            for k__ in keys(v, prefix=_full(k)):
                yield k__
        else:
            yield _full(k)


def config(attr):
    """
    Retrieves the queried attribute value from the config file.

    Loads the config file on first call.
    """
    if not hasattr(config, 'config'):
        with open(f'{PATH_BASE}/{DIR_PROJ}/config.json') as f:
            config.config = json.load(f)
    return get(config.config, attr)


def get_midi_paths(dnm):
    dset_dir = config(f'datasets.{dnm}.dir_nm')
    paths = sorted(glob.iglob(f'{PATH_BASE}/{DIR_DSET}/{dset_dir}/**/*.mid', recursive=True))
    return paths


MIDO_TPO = int(5e5)  # Midi default tempo (ms per tick, i.e. 120 BPM)


class MidoUtil:
    @staticmethod
    def get_msgs_by_type(midi, t_):
        def _get(track):
            return list(filter(lambda tr: tr.type == t_, track))
        if type(midi) is MidiFile:
            return {i: _get(trk) for i, trk in enumerate(midi.tracks)}
        else:  # midi.MidiTrack
            return _get(midi)

    @staticmethod
    def get_tempo_changes(midi):
        lst_msgs = MidoUtil.get_msgs_by_type(midi, 'set_tempo')
        # if msgs:
        #     return
        if lst_msgs:
            lst_msgs = list(filter(bool, lst_msgs.values()))
            lst_tempo = list(map(lambda lst: list(map(lambda msg: msg.tempo, lst)), lst_msgs))
            return flatten(lst_tempo)
        # ic(lst)
        else:
            return [MIDO_TPO]


class PrettyMidiUtil:
    INSTR_ATTRS = ('start', 'end', 'pitch', 'velocity')

    @staticmethod
    def plot_single_instrument(instr_, cols=INSTR_ATTRS, n=None):
        """
        Plots a single `pretty_midi.Instrument` channel

        :return: pd.DataFrame of respective attributes in the note, up until `n`
        """
        def _get_get(note):
            return [getattr(note, attr) for attr in cols]
        # df = pd.DataFrame([[n.start, n.end, n.pitch, n.velocity] for n in instr_.notes], columns=cols)
        df = pd.DataFrame([_get_get(n) for n in instr_.notes], columns=cols)[:n]
        df.plot(figsize=(16, 9), lw=0.25, ms=0.3, title=f'Instrument notes plot - [{",".join(cols)}]')
        return df
