import json
import pickle
import glob
from math import floor, ceil
from functools import reduce

import numpy as np
import pandas as pd
import mido
from mido import MidiFile
import pretty_midi
from pretty_midi import PrettyMIDI
import librosa
from librosa import display
import music21
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from icecream import ic

from data_path import *


# rcParams['figure.constrained_layout.use'] = True
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


DEF_TPO = int(5e5)  # Midi default tempo (ms per beat, i.e. 120 BPM)
N_NOTE_OCT = 12  # Number of notes in an octave


def tempo2bpm(tempo):
    """
    :param tempo: ms per beat
    :return: Beat per minute
    """
    return 60 * 1e6 / tempo


class MidoUtil:
    @staticmethod
    def get_msgs_by_type(fl, t_, as_dict=False):
        def _get(track):
            return list(filter(lambda tr: tr.type == t_, track))
        if type(fl) is MidiFile or type(fl) is str:
            if type(fl) is str:
                fl = MidiFile(fl)
            d_msgs = {i: _get(trk) for i, trk in enumerate(fl.tracks)}
            return d_msgs if as_dict else flatten(d_msgs.values())
        elif type(fl) is mido.MidiTrack:
            return _get(fl)

    @staticmethod
    def get_tempo_changes(mf, dedupe=False):
        lst_msgs = MidoUtil.get_msgs_by_type(mf, 'set_tempo')
        if lst_msgs:
            tempos = [msg.tempo for msg in lst_msgs]
            return list(set(tempos)) if dedupe else tempos
        else:
            return [DEF_TPO]


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
        df = pd.DataFrame([_get_get(n) for n in instr_.notes], columns=cols)[:n]
        df.plot(
            figsize=(16, 9),
            lw=0.25, ms=0.3,
            title=f'Instrument notes plot - [{",".join(cols)}]'
        )
        return df

    @staticmethod
    def get_pitch_range(pm_, clip=False):
        """
        :return: Inclusive lower and upper bound of pitch in PrettyMIDI
        """
        def _get_pitch_range():
            def _get(instr_):
                arr = np.array([n.pitch for n in instr_.notes])
                return np.array([arr.min(), arr.max()])

            if type(pm_) is PrettyMIDI:
                ranges = np.vstack([_get(i) for i in pm_.instruments])
                return ranges[:, 0].min(), ranges[:, 1].max()
            else:
                return list(_get(pm_))
        strt, end = _get_pitch_range()
        if clip:
            strt = floor(strt / N_NOTE_OCT) * N_NOTE_OCT
            end = ceil(end / N_NOTE_OCT) * N_NOTE_OCT
        return strt, end

    @staticmethod
    def plot_piano_roll(fl, strt=None, end=None, fqs=100, with_beats=True):
        """
        :param fl: A PrettyMIDI, Instrument, or file path string
        :param strt: Lowest pitch to plot inclusive, inferred if not given
        :param end: Highest pitch to plot inclusive, inferred if not given
        :param fqs: Sample frequency
        :param with_beats: If true, beat times are shown as vertical lines
            If fl is Instrument, expect the times needs to be provided
        """
        if type(fl) is str:
            fl = PrettyMIDI(fl)
        if strt is None and end is None:
            strt, end = PrettyMidiUtil.get_pitch_range(fl, clip=True)
            end += 1  # inclusive np array slicing
        pr_ = fl.get_piano_roll(fqs)[strt:end]
        strt_ = pretty_midi.note_number_to_name(strt)
        end_ = pretty_midi.note_number_to_name(end)

        plt.figure(figsize=(16, (9 * (end-strt+1) / 128) + 1), constrained_layout=True)
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
        nms = np.arange(strt, end, 6)
        plt.colorbar(pad=2**(-5))
        plt.yticks(
            list(map(pretty_midi.note_number_to_hz, nms)),
            list(map(pretty_midi.note_number_to_name, nms))
        )
        if (type(with_beats) is bool and with_beats) or with_beats is not None:
            ticks = plt.yticks()[0]
            x = fl.get_beats() if type(fl) is PrettyMIDI else with_beats
            ax = plt.gca()
            ax.vlines(x=x, ymin=ticks.min(), ymax=ticks.max(), lw=0.25, alpha=0.5, label='Beats')

        plt.title(f'Piano roll plot - [{strt_}, {end_}]')
        plt.show()


def eg_songs(k=None, pretty=False, fmt='MIDI'):
    """
    :return: A list of or single MIDI file path
    """
    dnm = f'{fmt}_EG'
    d_dset = config(f'{DIR_DSET}.{dnm}')
    dir_nm = d_dset['dir_nm']
    path = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'
    mids = sorted(glob.iglob(f'{path}/{d_dset["fmt_song"]}', recursive=True))
    if k:
        if type(k) is int:
            return mids[k]
        else:  # Expect str
            return next(filter(lambda p: p.find(k) != -1, mids))
    else:
        return [p[p.find(dir_nm):] for p in mids] if pretty else mids


if __name__ == '__main__':
    from icecream import ic
    # ic(tempo2bpm(DEF_TPO))

    def check_note2hz():
        for n in np.arange(0, 12*10, 12):
            ic(n, pretty_midi.note_number_to_hz(n))
    # check_note2hz()

    def check_piano_roll():
        # pm = pretty_midi.PrettyMIDI(eg_midis('Shape of You'))
        pm = pretty_midi.PrettyMIDI(eg_songs('Merry Go Round of Life'))

        pr = pm.get_piano_roll(100)
        # ic(pr.shape, pr.dtype, pr[75:80, 920:960])
        # # ic(np.where(pr > 100))
        #
        # instr0 = pm.instruments[0]
        # instr1 = pm.instruments[1]
        # ic(instr0.get_piano_roll()[76, 920:960])
        # ic(instr1.get_piano_roll()[76, 920:960])

        pmu = PrettyMidiUtil()
        pmu.plot_piano_roll(pm, fqs=100)
        # pmu.plot_piano_roll(instr0)
    # check_piano_roll()

    fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
    ic(fnm)
    scr = music21.converter.parse(fnm)
    # scr.plot('pianoroll', figureSize=(10, 3), constrained_layout=False)
    part: music21.stream.Part = scr.parts[0]
    ic(part)
    # r = part.plot('pianoroll')
    # ic(type(r), r)
    # r.run()
    # plt.show()
    # part.measures(numberStart=0, numberEnd=20).plot('pianoroll')
    # voice = scr.parts[0]
    # voice.measures(1, 10).show()

    from music21 import graph

    # colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    # data = []
    #
    # for numSharps in range(0, 7):
    #     keySig = key.KeySignature(numSharps)
    #     majScale = keySig.getScale('major')
    #     tonicPitch = majScale.tonic
    #     scaleDict = {'color': colors[numSharps]}
    #     for deg in range(1, 8):
    #         thisPitch = majScale.pitchFromDegree(deg)
    #         thisPitch.transposeAboveTarget(tonicPitch, inPlace=True)
    #         data.append((tonicPitch.pitchClass, thisPitch.pitchClass, thisPitch.frequency, scaleDict))
    #
    # ic(data[0:10])
    #
    # a = graph.primitives.Graph3DBars(title='Seven Major Scales',
    #                                  alpha=0.5,
    #                                  barWidth=0.2,
    #                                  useKeyValues=True,
    #                                  figureSize=(10, 10, 4),
    #                                  )
    # a.data = data
    # a.axis['x']['ticks'] = (range(12), ('c c# d d# e f f# g g# a a# b').split())
    # a.axis['y']['ticks'] = (range(12), ('c c# d d# e f f# g g# a a# b').split())
    # a.axis['z']['range'] = (0, 1000)
    #
    # a.setAxisLabel('x', 'Root Notes')
    # a.setAxisLabel('y', 'Scale Degrees')
    # a.setAxisLabel('z', 'Frequency in Hz')
    # a.process()


    def test_show_in_plot():
        data = [('Chopin', [(1810, 1849 - 1810)]),
                ('Schumanns', [(1810, 1856 - 1810), (1819, 1896 - 1819)]),
                ('Brahms', [(1833, 1897 - 1833)])]
        xTicks = [(1810, '1810'),
                  (1848, '1848'),
                  (1897, '1897')]
        ghb = graph.primitives.GraphHorizontalBar()
        ghb.title = 'Romantics live long and not so long'
        ghb.data = data
        ghb.setTicks('x', xTicks)

        ghb.doneAction = None  # The `show` action calls figure.show() which doesn't seem to work in Pycharm
        ghb.process()
        # ghb.callDoneAction()
        plt.show()
    # ghb.figure.show()

    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # fig = plt.figure()
    # x = np.arange(20) / 50
    # y = (x + 0.1) * 2
    #
    # val1 = [True, False] * 10
    # val2 = [False, True] * 10
    #
    # plt.errorbar(x, y,
    #              xerr=0.1,
    #              xlolims=True,
    #              label='Line 1')
    # y = (x + 0.3) * 3
    #
    # plt.errorbar(x + 0.6, y,
    #              xerr=0.1,
    #              xuplims=val1,
    #              xlolims=val2,
    #              label='Line 2')
    #
    # y = (x + 0.6) * 4
    # plt.errorbar(x + 1.2, y,
    #              xerr=0.1,
    #              xuplims=True,
    #              label='Line 3')
    #
    # plt.legend()
    #
    # fig.suptitle("""matplotlib.figure.Figure.show()
    # function Example\n\n""", fontweight="bold")
    #
    # fig.show()
