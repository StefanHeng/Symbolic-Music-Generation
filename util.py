import json
import pickle
import glob
from functools import reduce

import numpy as np
import pandas as pd
from mido import MidiFile
import pretty_midi
from pretty_midi import PrettyMIDI
import librosa
from librosa import display
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from data_path import *


rcParams['figure.constrained_layout.use'] = True
sns.set_style('darkgrid')


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
    # def __init__(self):
    #     pass

    @staticmethod
    def plot_single_instrument(instr_, cols=('start', 'end', 'pitch', 'velocity'), n=None):
        """
        Plots a single `pretty_midi.Instrument` channel

        :return: pd.DataFrame of respective attributes in the note, up until `n`
        """
        def _get_get(note):
            return [getattr(note, attr) for attr in cols]
        # df = pd.DataFrame([[n.start, n.end, n.pitch, n.velocity] for n in instr_.notes], columns=cols)
        df = pd.DataFrame([_get_get(n) for n in instr_.notes], columns=cols)[:n]
        df.plot(
            figsize=(16, 9),
            lw=0.25, ms=0.3,
            title=f'Instrument notes plot - [{",".join(cols)}]'
        )
        return df

    @staticmethod
    def get_pitch_range(pm_):
        """
        :return: Inclusive lower and upper bound of pitch in PrettyMIDI
        """
        def _get(instr_):
            arr = np.array([n.pitch for n in instr_.notes])
            return np.array([arr.min(), arr.max()])

        if type(pm_) is PrettyMIDI:
            ranges = np.vstack([_get(i) for i in pm_.instruments])
            return ranges[:, 0].min(), ranges[:, 1].max()
        else:
            return list(_get(pm_))

    @staticmethod
    def plot_piano_roll(pm_: pretty_midi.PrettyMIDI, strt=None, end=None, fqs=100):
        if strt is None and end is None:
            strt, end = PrettyMidiUtil.get_pitch_range(pm_)
            end += 1  # inclusive np array slicing
        pr_ = pm_.get_piano_roll(fqs)[strt:end]
        strt_ = pretty_midi.note_number_to_name(strt)
        end_ = pretty_midi.note_number_to_name(end)

        fig = plt.figure(figsize=(16, (9 * (end-strt+1) / 128) + 1))
        ax = fig.add_subplot(111)
        kwargs = dict(
            fmin=pretty_midi.note_number_to_hz(strt) if strt else None
        )
        with sns.axes_style('ticks'):
            librosa.display.specshow(
                pr_,
                hop_length=1, sr=fqs, x_axis='time', y_axis='cqt_note',
                cmap='mako',
                **kwargs
            )
        # nms = np.arange(strt, end, 5)
        # plt.yticks(nms, list(map(pretty_midi.note_number_to_name, nms)))
        plt.title(f'Piano roll plot - [{strt_}, {end_}]')
        plt.show()


if __name__ == '__main__':
    from icecream import ic

    def _get_pm():
        dnm = 'MIDI_EG'
        d_dset = config(f'{DIR_DSET}.{dnm}')
        dir_nm = d_dset['dir_nm']
        path = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'
        mids = sorted(glob.iglob(f'{path}/{d_dset["fmt_midi"]}', recursive=True))
        mid_eg = mids[2]

        return pretty_midi.PrettyMIDI(mid_eg)

    def check_piano_roll():
        pm = _get_pm()

        pr = pm.get_piano_roll(100)
        ic(pr.shape, pr.dtype, pr[75:80, 920:960])
        # ic(np.where(pr > 100))

        instr0 = pm.instruments[0]
        instr1 = pm.instruments[1]
        ic(instr0.get_piano_roll()[76, 920:960])
        ic(instr1.get_piano_roll()[76, 920:960])

        pmu = PrettyMidiUtil()
        pmu.plot_piano_roll(pm, fqs=100)
        # pmu.plot_piano_roll(instr0)

    check_piano_roll()
