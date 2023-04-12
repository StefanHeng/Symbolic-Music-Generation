import os
import pickle
from os.path import join as os_join
from typing import Dict, Tuple, Any, Optional
from collections import Counter

import pandas as pd
import music21 as m21
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


from stefutil import *
from musicnlp.util import *
import musicnlp.util.music as music_util


__all__ = ['get_plot_info']


logger = get_logger('MXL Check')


def extract_single(fnm: str = None) -> Optional[Dict[str, Any]]:
    try:
        scr = m21.converter.parse(fnm)
    except Exception as e:
        logger.warn(f'Failed to read piece {pl.i(stem(fnm, keep_ext=True))} w/ error {pl.i(e)}')
        return
    # grab all notes in the piece
    notes = scr.flat.notes
    rests = scr.flat[m21.note.Rest]
    return dict(
        n_notes=len(notes),
        n_rests=len(rests),
        n_time_sig=len(scr.flat[m21.meter.TimeSignature]),
        n_key_sig=len(scr.flat[m21.key.KeySignature]),
        n_tempo=len(scr.flat[m21.tempo.MetronomeMark]),
        n_bar=len(scr.parts[0][m21.stream.Measure]),
        durations_note=dict(Counter(n.duration.quarterLength for n in notes)),
        durations_rest=dict(Counter(n.duration.quarterLength for n in rests))
    )


def extract(dataset_name: str = None):
    read_errs_ = dict(mid=0, mxl=0)

    rows = []
    concurrent = True
    mic(concurrent)
    for kd in ['mid', 'mxl']:
        fnms = music_util.get_converted_song_paths(dataset_name=dataset_name, fmt=kd)
        desc = f'Extracting {kd} info'
        if concurrent:
            args = dict(mode='process', batch_size=None, with_tqdm=dict(desc=desc, total=len(fnms)))
            for d in conc_yield(fn=extract_single, args=fnms, **args):  # Doesn't work in ipython notebook
                if d is None:
                    read_errs_[kd] += 1
                else:
                    rows.append(d | dict(format=kd))
        else:
            for f in tqdm(fnms, desc=desc):
                d = extract_single(f)
                if d is None:
                    read_errs_[kd] += 1
                else:
                    rows.append(d | dict(format=kd))
    return pd.DataFrame(rows), read_errs_


def get_plot_info(dataset_name: str = None, cache: str = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if cache:
        path = os_join(u.proj_path, 'notebook', 'pre_process', f'{cache}.pkl')
        if os.path.exists(path):
            logger.info(f'Loading cached data from {pl.i(path)}... ')
            with open(path, 'rb') as fl:
                c = pickle.load(fl)
            df, read_errs = c['df'], c['read_errs']
        else:
            df, read_errs = extract(dataset_name=dataset_name)
            with open(path, 'wb') as fl:
                pickle.dump(dict(df=df, read_errs=read_errs), fl)
                logger.info(f'Cached data saved to {pl.i(path)} ')
    else:
        df, read_errs = extract(dataset_name=dataset_name)
    return df, read_errs


if __name__ == '__main__':
    dnm = 'POP909'
    cch = f'Mxl-Check-Cache_{dnm}'

    def check_run():
        df, read_errs = get_plot_info(dataset_name=dnm, cache=cch)
        mic(df, read_errs)
    check_run()
