import os
import glob
import json
import pathlib
import pickle
from math import floor, ceil
from functools import reduce
import itertools
import concurrent.futures
from typing import TypeVar, Callable, Union
from collections.abc import Iterable
import datetime

import colorama
import numpy as np
import pandas as pd
import mido
from mido import MidiFile
import pretty_midi
from pretty_midi import PrettyMIDI
import librosa
from librosa import display
import music21 as m21
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

from data_path import *


rcParams['figure.constrained_layout.use'] = True
sns.set_style('darkgrid')

nan = float('nan')


def flatten(lsts):
    """ Flatten list of [list of elements] to list of elements """
    return sum(lsts, [])


def clip(val, vmin, vmax):
    return max(min(val, vmax), vmin)


def np_index(arr, idx):
    return np.where(arr == idx)[0][0]


def vars_(obj, include_private=False):
    """
    :return: A variant of `vars` that returns all properties and corresponding values in `dir`, except the
    generic ones that begins with `_`
    """
    def is_relevant():
        if include_private:
            return lambda a: not a.startswith('__')
        else:
            return lambda a: not a.startswith('__') and not a.startswith('_')
    attrs = filter(is_relevant(), dir(obj))
    return {a: getattr(obj, a) for a in attrs}


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


def conc_map(fn, it):
    """
    Wrapper for `concurrent.futures.map`

    :param fn: A function
    :param it: A list of elements
    :return: Iterator of `lst` elements mapped by `fn` with concurrency
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return executor.map(fn, it)


def now(as_str=True):
    d = datetime.datetime.now()
    return d.strftime('%Y-%m-%d %H:%M:%S') if as_str else d


def fmt_dt(secs: Union[int, float, datetime.timedelta]):
    if isinstance(secs, datetime.timedelta):
        secs = secs.seconds + (secs.microseconds/1e6)
    if secs >= 86400:
        d = secs // 86400  # // floor division
        return f'{round(d)}d{fmt_dt(secs-d*86400)}'
    elif secs >= 3600:
        h = secs // 3600
        return f'{round(h)}h{fmt_dt(secs-h*3600)}'
    elif secs >= 60:
        m = secs // 60
        return f'{round(m)}m{fmt_dt(secs-m*60)}'
    else:
        return f'{round(secs)}s'


T = TypeVar('T')


def compress(lst: list[T]) -> list[tuple[T, int]]:
    """
    :return: A compressed version of `lst`, as 2-tuple containing the occurrence counts
    """
    if not lst:
        return []
    return ([(lst[0], len(list(itertools.takewhile(lambda elm: elm == lst[0], lst))))]
            + compress(list(itertools.dropwhile(lambda elm: elm == lst[0], lst))))


def split(lst: list[T], call: Callable[[T], bool]) -> list[list[T]]:
    """
    :return: Split a list by locations of elements satisfying a condition
    """
    return [list(g) for k, g in itertools.groupby(lst, call) if k]


def join_its(its: Iterable[Iterable[T]]) -> Iterable[T]:
    out = itertools.chain()
    for it in its:
        out = itertools.chain(out, it)
    return out


def log(s, c: str = 'log', c_time='green', as_str=False):
    """
    Prints `s` to console with color `c`
    """
    if not hasattr(log, 'reset'):
        log.reset = colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL
    if not hasattr(log, 'd'):
        log.d = dict(
            log='',
            warn=colorama.Fore.YELLOW,
            error=colorama.Fore.RED,
            err=colorama.Fore.RED,
            success=colorama.Fore.GREEN,
            suc=colorama.Fore.GREEN,
            info=colorama.Fore.BLUE,
            i=colorama.Fore.BLUE,
            w=colorama.Fore.RED,

            y=colorama.Fore.YELLOW,
            yellow=colorama.Fore.YELLOW,
            red=colorama.Fore.RED,
            r=colorama.Fore.RED,
            green=colorama.Fore.GREEN,
            g=colorama.Fore.GREEN,
            blue=colorama.Fore.BLUE,
            b=colorama.Fore.BLUE,
        )
    if c in log.d:
        c = log.d[c]
    if as_str:
        return f'{c}{s}{log.reset}'
    else:
        print(f'{c}{log(now(), c=c_time, as_str=True)}| {s}{log.reset}')


def logs(s, c):
    return log(s, c=c, as_str=True)


def logi(s):
    """
    Syntactic sugar for logging `info` as string
    """
    return logs(s, c='i')


def config(attr):
    """
    Retrieves the queried attribute value from the config file.

    Loads the config file on first call.
    """
    if not hasattr(config, 'config'):
        with open(f'{PATH_BASE}/{DIR_PROJ}/config.json') as f:
            config.config = json.load(f)
            # str -> int
            for k, v in config.config['Melody-Extraction']['tokenizer'].items():
                if isinstance(v, dict):
                    config.config['Melody-Extraction']['tokenizer'][k] = {
                        (int(k_) if k_.isnumeric() else k_): v_
                        for k_, v_ in v.items()
                    }
    return get(config.config, attr)


DEF_TPO = int(5e5)  # Midi default tempo (ms per beat, i.e. 120 BPM)
N_NOTE_OCT = 12  # Number of notes in an octave


def tempo2bpm(tempo):
    """
    :param tempo: ms per beat
    :return: Beat per minute
    """
    return 60 * 1e6 / tempo


class MidoUtil:
    def __init__(self):
        pass

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
    def __init__(self):
        pass

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
    def get_pitch_range(pm_, clip_=False):
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
        if clip_:
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
            strt, end = PrettyMidiUtil.get_pitch_range(fl, clip_=True)
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


class Music21Util:
    def __init__(self):
        pass

    @staticmethod
    def plot_piano_roll(stream, title=None, s=0, e=None):
        """
        :return: music21 graph object for plotting
        """
        mess = stream.measures(s, e)
        plt_ = m21.graph.plot.HorizontalBarPitchSpaceOffset(
            mess, figsize=(16, 9), constrained_layout=False, doneAction=None
        )
        plt_.colors = sns.color_palette(palette='husl', n_colors=2**4)
        plt_.fontFamily = 'sans-serif'
        plt_.run()

        plt.tight_layout()
        x_ticks = plt.xticks()[0]
        y_ticks = plt.yticks()[0]
        x_d, y_d = np.diff(x_ticks).max(), np.diff(y_ticks).max()
        offset_x, offset_y = x_d / 2**3, y_d / 2**1
        x_s, x_e = x_ticks[0], x_ticks[-1] + x_d
        y_s, y_e = y_ticks[0], y_ticks[-1]
        plt.xlim([x_s-offset_x, x_e+offset_x])
        plt.ylim([y_s-offset_y, y_e+offset_y])

        fig = plt.gcf()
        fig.set_size_inches(16, clip(9 * (y_e-y_s) / (x_e-x_s), 16/2**3, 16/2))
        strt = s
        end = len(stream.measures(0, e)) if e is None else e
        title = (
            title or
            (stream.metadata and stream.metadata.title) or
            (stream.activeSite and stream.activeSite.metadata and stream.activeSite.metadata.title) or
            stream.id
        )
        if isinstance(stream, m21.stream.Part) and stream.partName:
            title = f'{title}, {stream.partName}'
        plt.title(f'Piano roll, {title}, bars {strt}-{end}')
        txt_prop = dict(fontname='DejaVu Sans')
        plt.yticks(*plt.yticks(), **txt_prop)
        plt.show()
        return plt_


def eg_songs(k=None, pretty=False, fmt='MIDI'):
    """
    :return: A list of or single MIDI file path
    """
    dnm = f'{fmt}_EG'
    d_dset = config(f'{DIR_DSET}.{dnm}')
    dir_nm = d_dset['dir_nm']
    path = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'
    mids = sorted(glob.iglob(f'{path}/{d_dset["song_fmt"]}', recursive=True))
    if k:
        if type(k) is int:
            return mids[k]
        else:  # Expect str
            return next(filter(lambda p: p.find(k) != -1, mids))
    else:
        return [p[p.find(dir_nm):] for p in mids] if pretty else mids


def fl_nms(dnm, k='song_fmt') -> list[str]:
    """
    :return: List of music file paths
    """
    if not hasattr(fl_nms, 'd_dsets'):
        fl_nms.d_dsets = config('datasets')
    d_dset = fl_nms.d_dsets[dnm]
    return sorted(
        glob.iglob(os.path.join(PATH_BASE, DIR_DSET, d_dset['dir_nm'], d_dset[k]), recursive=True)
    )


def stem(path, ext=False):
    """
    :param path: A potentially full path to a file
    :param ext: If True, file extensions is preserved
    :return: The file name, without parent directories
    """
    return os.path.basename(path) if ext else pathlib.Path(path).stem


if __name__ == '__main__':
    from icecream import ic
    from music21 import graph

    # ic(config('Melody-Extraction.tokenizer'))

    # ic(tempo2bpm(DEF_TPO))

    def check_note2hz():
        for n in np.arange(0, 12*10, 12):
            ic(n, pretty_midi.note_number_to_hz(n))
    # check_note2hz()

    def check_piano_roll():
        # pm = pretty_midi.PrettyMIDI(eg_midis('Shape of You'))
        pm = pretty_midi.PrettyMIDI(eg_songs('Merry Go Round of Life'))

        # pr = pm.get_piano_roll(100)
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
    # test_show_in_plot()

    def test_piano_roll():
        fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        ic(fnm)

        scr = m21.converter.parse(fnm)
        ic(scr.id)
        # Looks like this doesn't work **sometimes**?
        # r = scr.plot('pianoroll', figureSize=(16, 9), doneAction=None)
        # ic(r)

        part = scr.parts[0]
        # plt_ = graph.plot.HorizontalBarPitchSpaceOffset(part.measures(25, 90), doneAction=None, figsize=(16, 9))
        # plt_.run()
        # ic(len(plt_.data), plt_.data)
        # plt.tight_layout()
        # ic(plt.xlim(), plt.ylim())
        # ic(plt.xticks(), plt.yticks())
        # # plt.xlim([0, 501])
        # plt.show()

        m2u = Music21Util()
        # ms = part.measures(20, 100)
        # m2u.plot_piano_roll(ms, s=10, e=30)
        # m2u.plot_piano_roll(scr, s=10, e=15)
        m2u.plot_piano_roll(part, s=20, e=40)
        # ic(type(ms), vars(ms), dir(ms))
        # ic(ms.measures(26, 30))
        # ic(mes, nums)
        # for mes in ms.measures(0, None):
        #     ic(mes)
    # test_piano_roll()

    def check_show_title():
        fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        ic(fnm)
        scr = m21.converter.parse(fnm)
        ic(scr)
        # ic(len(dir(scr)))
        # ic(vars_(scr, include_private=False))
        meta = scr.metadata
        # ic(meta, vars(meta), vars_(meta))
        ic(meta.title, meta.composer)
        part_ch2 = scr.parts[1]
        ic(part_ch2, part_ch2.partName, part_ch2.metadata)
        # ic(vars(part_ch2), vars_(part_ch2))
        ic(part_ch2.activeSite.metadata.title)
    # check_show_title()

    def check_compress():
        arr = [
            202, 202, 202, 202, 203, 203, 203, 203, 202, 202, 202, 202, 203,
            203, 203, 203, 202, 202, 202, 202, 203, 203, 203, 203
        ]
        ic(compress(arr))
    # check_compress()

    def check_fl_nms():
        dnm = 'POP909'
        fnms = fl_nms(dnm)
        ic(len(fnms), fnms[:20])
        fnms = fl_nms(dnm, k='song_fmt_exp')
        ic(len(fnms), fnms[:20])
    check_fl_nms()

    def setup_pop909():
        from shutil import copyfile
        from tqdm import trange
        dnm = 'POP909'
        path = os.path.join(PATH_BASE, DIR_DSET, 'POP909-Dataset', dnm)
        path_exp = os.path.join(PATH_BASE, DIR_DSET, dnm)
        os.makedirs(path_exp, exist_ok=True)

        df = pd.read_excel(os.path.join(path, 'index.xlsx'))
        paths = sorted(glob.iglob(os.path.join(path, '*/*.mid'), recursive=True))
        for i in trange(len(paths)):
            p = paths[i]
            rec = df.iloc[i, :]
            fnm = f'{rec["artist"]} - {rec["name"]}.mid'
            copyfile(p, os.path.join(path_exp, fnm))
    # setup_pop909()
