import os
import re
import pickle
from os.path import join as os_join
from typing import Dict, Tuple, List, Any, Optional, Iterable, Union
from fractions import Fraction
from collections import Counter

import numpy as np
import pandas as pd
import music21 as m21
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


from stefutil import *
from musicnlp.util import *
import musicnlp.util.music as music_util


__all__ = ['get_plot_info', 'split_df_by_dataset', 'side_by_side_plot']


logger = get_logger('MXL Check')


def extract_single(fnm: str = None) -> Optional[Dict[str, Any]]:
    try:
        scr = m21.converter.parse(fnm)
    except Exception as e:
        logger.warning(f'Failed to read piece {pl.i(stem(fnm, keep_ext=True))} w/ error {pl.i(e)}')
        return
    # grab all notes in the piece
    notes = scr.flat.notes
    rests = scr.flat[m21.note.Rest]
    return dict(
        n_note=len(notes),
        n_rest=len(rests),
        n_time_sig=len(scr.flat[m21.meter.TimeSignature]),
        n_key_sig=len(scr.flat[m21.key.KeySignature]),
        n_tempo=len(scr.flat[m21.tempo.MetronomeMark]),
        n_bar=len(scr.parts[0][m21.stream.Measure]),
        durations_note=dict(Counter(n.duration.quarterLength for n in notes)),
        durations_rest=dict(Counter(n.duration.quarterLength for n in rests))
    )


def extract(dataset_names: List[str] = None):
    fmts = [f'{dnm}-{kd}' for dnm in dataset_names for kd in ['mid', 'mxl']]
    read_errs_ = {f: 0 for f in fmts}

    rows = []
    concurrent = True
    for dnm in dataset_names:
        for kd in ['mid', 'mxl']:
            fnms = music_util.get_converted_song_paths(dataset_name=dnm, fmt=kd, backend='all')
            desc = f'Extracting {dnm} {kd} info'
            fmt = f'{dnm}-{kd}'
            if concurrent:
                # No-batching is faster for POP909
                args = dict(mode='process', batch_size=None, with_tqdm=dict(desc=desc, total=len(fnms)))
                for d in conc_yield(fn=extract_single, args=fnms, **args):  # Doesn't work in ipython notebook
                    if d is None:
                        read_errs_[fmt] += 1
                    else:
                        rows.append(d | dict(format=fmt))
            else:
                for f in tqdm(fnms, desc=desc):
                    d = extract_single(f)
                    if d is None:
                        read_errs_[fmt] += 1
                    else:
                        rows.append(d | dict(format=fmt))
    return pd.DataFrame(rows), read_errs_


def get_plot_info(dataset_names: List[str] = None, cache: str = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if cache:
        path = os_join(u.proj_path, 'notebook', 'pre_process', f'{cache}.pkl')
        if os.path.exists(path):
            logger.info(f'Loading cached data from {pl.i(path)}... ')
            with open(path, 'rb') as fl:
                c = pickle.load(fl)
            df, read_errs = c['df'], c['read_errs']
        else:
            df, read_errs = extract(dataset_names=dataset_names)
            with open(path, 'wb') as fl:
                pickle.dump(dict(df=df, read_errs=read_errs), fl)
                logger.info(f'Cached data saved to {pl.i(path)} ')
    else:
        df, read_errs = extract(dataset_names=dataset_names)
    return df, read_errs


format_pattern = re.compile(r'^(?P<dataset_name>\w+)-(?P<fmt>\w+)$')
dnm_order = ['POP909', 'MAESTRO', 'LMD']


def split_df_by_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    fmts = df.format.unique()
    dnms_ = set(format_pattern.match(f).group('dataset_name') for f in fmts)
    dnms_ = sorted(dnms_, key=lambda dnm: dnm_order.index(dnm))
    return {dnm: df[df.format.str.startswith(dnm)] for dnm in dnms_}


def merge_counts(cs: Iterable[Dict]) -> Dict:
    c = Counter()
    for d in cs:
        c.update(d)
    return c


def merge_counter_data(df: pd.DataFrame = None, aspect: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert df[aspect].apply(lambda d: isinstance(d, dict)).all()
    df = df[[aspect, 'format']]

    # group by format, for each format, create a Counter and update by dict
    srs_counts = df.groupby('format').apply(lambda d: d[aspect].apply(Counter).sum())
    fmt_ = next(iter(df['format'].unique()))
    assert srs_counts[fmt_] == merge_counts(df[df.format == fmt_][aspect])   # sanity check pd operation
    # for each counter, expand it to `aspect`, `count` pairs
    dfs = []
    for fmt, c in srs_counts.items():
        df_ = pd.DataFrame.from_dict(c, orient='index', columns=['count']).reset_index()
        df_['format'] = fmt
        dfs.append(df_)
    ret = pd.concat(dfs, ignore_index=True)
    ret.rename(columns={'index': aspect}, inplace=True)

    # if there are any Fractions, they are durations, make each unique duration a discrete category for histogram
    assert ret[aspect].apply(lambda d: isinstance(d, Fraction)).any()

    # convert potential Fractions to float
    ret_float, ret_str = ret.copy(), ret.copy()
    ret_float[aspect] = ret[aspect].apply(lambda d: float(d) if isinstance(d, Fraction) else d)

    # convert to string categories
    cats = [str(d) for d in sorted(ret_str[aspect].unique(), key=float)]
    ret_str[aspect] = ret_str[aspect].map(str)
    df_col2cat_col(ret_str, aspect, categories=cats)
    return ret_str, ret_float


def expand_by_count(df: pd.DataFrame, count_col: str) -> pd.DataFrame:
    df = df.reindex(df.index.repeat(df[count_col]))
    df.reset_index(drop=True, inplace=True)
    return df


def side_by_side_plot(
        df: pd.DataFrame = None, aspect: str = None, hist_args: Dict[str, Any] = None, box_args: Dict[str, Any] = None,
        upper_percentile: Union[float, int] = None, subplots_args: Dict[str, Any] = None, title: str = None
):
    is_counter_data = df[aspect].apply(lambda d: isinstance(d, dict)).all()

    df_hs, df_bx = df, df
    if is_counter_data:
        df_hs, df_bx = merge_counter_data(df=df, aspect=aspect)
        wt = 'count'
        df_bx = expand_by_count(df=df_bx, count_col='count')
    else:  # sanity check
        wt = None
        assert df[aspect].apply(lambda d: isinstance(d, (int, float))).all()
    fig, axs = plt.subplots(nrows=2, ncols=1, **(subplots_args or dict()))
    args = dict(stat='percent', kde=True, palette='husl', weights=wt)
    # check if `aspect` contains all the same value
    if df_hs[aspect].nunique() == 1:
        args['kde'] = False
    args |= (hist_args or dict())
    sns.histplot(data=df_hs, x=aspect, hue='format', ax=axs[0], **args)
    if upper_percentile:
        mas = []
        for fmt in df_hs.format.unique():
            df_ = df_hs[df_hs.format == fmt][aspect]
            mas.append(np.percentile(df_, upper_percentile))
        axs[0].set_xlim(left=0, right=max(mas))

    # TODO: change color
    od_gray = hex2rgb('#282C34', normalize=True)
    mp = dict(marker='s', markerfacecolor='white', markeredgecolor=od_gray)
    args = dict(showmeans=True, meanprops=mp, palette='husl') | (box_args or dict())
    sns.boxplot(data=df_bx, x='format', y=aspect, ax=axs[1], **args)

    title_ = f'MID v.s. MXL on {aspect}'
    if title:
        title_ = f'{title_} ({title})'
    fig.suptitle(title_)

    if is_counter_data:
        ax = axs[0]
        ax.xaxis.set_tick_params(labelsize=7)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        x_ticks = ax.get_xticks()
        mi, ma = min(x_ticks), max(x_ticks)
        ax.set_xlim([mi - 0.5, ma + 0.5])  # Reduce the white space on both sides

        counts = df_hs
    else:
        counts = df.groupby('format').apply(lambda d: d[aspect].value_counts())
    return axs, counts


if __name__ == '__main__':
    pd.set_option('display.max_rows', 256)

    # dnms = ['POP909']
    dnms = ['POP909', 'MAESTRO']
    # dnms = ['POP909', 'MAESTRO', 'LMD']
    if dnms == ['POP909']:
        cnm = f'Mxl-Check-Cache_{{dnm=pop}}'
    elif dnms == ['POP909', 'MAESTRO']:
        cnm = f'Mxl-Check-Cache_{{dnm=pop&mst}}'
    else:
        cnm = f'Mxl-Check-Cache_{{dnm=all}}'

    def check_file_loading():
        dnm = 'POP909'
        fnms_xml = music_util.get_converted_song_paths(dataset_name=dnm, fmt='mxl')
        fnms_midi = music_util.get_converted_song_paths(dataset_name=dnm, fmt='mid')
        assert all(stem(fn) == stem(fn2) for fn, fn2 in zip(fnms_xml, fnms_midi))  # sanity check files are paired
        mic(fnms_xml[:3], fnms_midi[:3])
    # check_file_loading()

    def check_run():
        df, read_errs = get_plot_info(dataset_names=dnms, cache=cnm)
        mic(df, read_errs)

        # k = 'n_note'
        # k = 'durations_note'
        k = 'n_tempo'

        for dnm, df_ in split_df_by_dataset(df=df).items():
            if dnm == 'POP909':
                continue
            mic(df_)
            _, counts = side_by_side_plot(df=df_, aspect=k, title=dnm)
            mic(dnm, counts)
            plt.show()
        # save_fig(f'{dnm}_{k}')
    check_run()
