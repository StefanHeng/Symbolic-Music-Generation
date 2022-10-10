import re
import glob
import itertools
import statistics
from os.path import join as os_join
from typing import List, Tuple, Dict, Sequence, Any, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.core.util import event_pb2
import matplotlib.pyplot as plt
import seaborn as sns

from stefutil import *
from musicnlp.util import *


def parse_tensorboard(path) -> Dict[str, pd.DataFrame]:
    """
    Modified from https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/

    parse a tensor board, only 1 tag supported, each time step should have the same fixed number of values
    """
    def parse_single(evt):
        assert len(evt.summary.value) == 1
        return dict(
            wall_time=evt.wall_time,
            name=evt.summary.value[0].tag,
            step=evt.step,
            value=float(evt.summary.value[0].simple_value),
        )

    fnms = list(glob.iglob(os_join(path, '**/events.out.tfevents*'), recursive=True))
    assert len(fnms) == 1, f'Expect one events.out.tfevents file, found {pl.i(len(fnms))}'
    fnm = fnms[0]
    events = [event_pb2.Event.FromString(rec.numpy()) for rec in tf.data.TFRecordDataset(fnm)]
    events = [parse_single(e) for e in events if len(e.summary.value)]
    events.sort(key=lambda e: (e['step'], e['wall_time']))
    events = [list(v) for k, v in itertools.groupby(events, key=lambda e: e['step'])]

    pattern_name = re.compile(r'(?P<tag>.*)/(?P<key>.*)')

    def name2tag_n_key(name: str) -> Tuple[str, str]:
        m = pattern_name.match(name)
        return m.group('tag'), m.group('key')

    def group_single(group_events: List[Dict]) -> Dict[str, Dict[str, Any]]:  # expects certain formatting of the `name`
        d_out = defaultdict(dict)
        for e in group_events:
            tag, key = name2tag_n_key(e['name'])
            if 'step' not in d_out[tag]:
                d_out[tag]['step'] = e['step']
            d_out[tag][key] = e['value']
        return d_out
    events = [group_single(e) for e in events]
    tags = set(tag for e in events for tag in e.keys())
    assert tags == {'train', 'eval'}, f'Expect {pl.i("train")} and {pl.i("eval")} tags, found {pl.i(tags)}'
    d_dfs = {tag: pd.DataFrame([d_out[tag] for d_out in events if tag in d_out]) for tag in tags}
    for tag, df_ in d_dfs.items():
        mi, ma = df_.step.min(), df_.step.max()
        assert np.array_equal(df_.step.to_numpy(), np.arange(mi, ma + 1)), \
            f'Expect step for {pl.i(tag)} to be continuously increasing integer range'
    return d_dfs


def parse_tensorboards(paths: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Parse multiple tensorboard files & merge them by steps
        Intended for resuming training from checkpoints
    :param paths: Paths to tensorboard files, in increasing order of ste[s
    """
    lst_d_dfs = [parse_tensorboard(path) for path in paths]
    tags = lst_d_dfs[0].keys()
    assert all(d_dfs.keys() == tags for d_dfs in lst_d_dfs), \
        f'Expect all tensorboard files to have the same tags, found {pl.i(tags)}'
    d_df_out = dict()
    for tag in tags:
        dfs = [d_dfs[tag] for d_dfs in lst_d_dfs]
        mis, mas = [df.step.min() for df in dfs], [df.step.max() for df in dfs]
        mis_correct = all(mi < mis[i+1] for i, mi in enumerate(mis[:-1]))
        mas_correct = all(ma < mas[i+1] for i, ma in enumerate(mas[:-1]))
        assert mis_correct and mas_correct, 'Steps for each consecutive tensorboard should be increasing'
        dfs_ = [dfs[-1]]  # add dataframes backwards, earlier overlapping data will be truncated
        for i, df_ in reversed(list(enumerate(dfs[:-1]))):
            dfs_.append(df_[df_.step < mis[i+1]])
        d_df_out[tag] = pd.concat(reversed(dfs_)).reset_index(drop=True)
    return d_df_out


def smooth(vals: Sequence[float], factor: float) -> np.array:
    last = vals[0]
    smoothed = np.empty(len(vals), dtype=np.float32)
    for i, v in enumerate(vals):
        last = smoothed[i] = factor*last + (1-factor) * v
    return smoothed


def plot_tb(
        d_df: Dict[str, pd.DataFrame], y: str = 'loss', plot_label: str = None,
        xlabel: str = None, ylabel: str = None,
        splits: Union[Tuple[str], str] = ('train', 'eval'), prefix_split: bool = True,
        smooth_factor: Union[float, Dict[str, float]] = 0.9, figure_kwargs: Dict = None,
        title: str = None, cs=None, save=False, show=True, ax=None,
        steps_per_epoch: int = None, smoothed_only: bool = False,
):
    ylabel = ylabel if ylabel is not None else y
    xlabel = xlabel if xlabel is not None else 'step'
    plot_label = plot_label if plot_label is not None else y
    assert all(k in ['train', 'eval'] for k in d_df.keys())
    if not ax:
        plt.figure(**(figure_kwargs or dict()))
    ax = ax or plt.gca()
    cs = cs or sns.color_palette(palette='husl', n_colors=7)

    def plot_single(idx_, tag_):
        df = d_df[tag_]
        x = df.step
        if tag_ == 'eval' and steps_per_epoch:
            x *= steps_per_epoch  # to match the time step for training, see `util.train_util_wrap.py`
        y_ = df[y]
        if 'acc' in y:
            y_ *= 100
        factor = smooth_factor[tag_] if isinstance(smooth_factor, dict) else smooth_factor
        y_s = smooth(y_, factor=factor)
        c = cs[idx_]
        ms = statistics.harmonic_mean(plt.gcf().get_size_inches()) / 16
        args_ori = LN_KWARGS | dict(ls='None', c=c, alpha=0.3, ms=ms)
        if tag_ == 'eval':  # enlarge markers for eval
            args_ori |= dict(ms=ms * 16, marker='1', alpha=0.9)
        args_smooth = LN_KWARGS | dict(c=c, lw=0.75, marker=None)
        if not smoothed_only:
            ax.plot(x, y_, **args_ori)
        return ax.plot(x, y_s, **args_smooth, label=f'{tag_} {plot_label}' if prefix_split else plot_label)
    plt.xlabel(xlabel)
    splits = splits if isinstance(splits, (tuple, list)) else (splits,)
    for idx, tag in enumerate(splits):
        plot_single(idx, tag)
        plt.ylabel(ylabel)
        plt.legend()
    save_title = 'Training per-batch performance over Steps'
    if title is None:
        title = save_title
    if title != 'none':
        plt.title(title)
        save_title = title
    if save:
        save_fig(f'{save_title}, {now(for_path=True)}')
    if show:
        plt.show()


def md_n_dir2tb_path(model_name: str = 'reformer', directory_name: str = None) -> str:
    return os_join(BASE_PATH, PROJ_DIR, MODEL_DIR, model_name, directory_name)


if __name__ == '__main__':
    # path_ = '/Users/stefanh/Documents/UMich/Research/Music with NLP/Symbolic-Music-Generation/models'
    # directory_name = '2022-04-01_09-40-48'

    def check_plot_single():
        df = parse_tensorboard(md_n_dir2tb_path(directory_name='2022-04-03_00-20-53'))
        mic(df)
        plot_tb(df, y='ntp_acc', save=False)
    # check_plot_single()

    def check_plot_multiple():
        # dir_nms = ['2022-04-03_00-20-53', '2022-04-03_11-01-04']
        dir_nms = ['2022-04-15_13-42-56', '2022-04-15_18-45-49', '2022-04-16_16-08-03']
        paths = [md_n_dir2tb_path(directory_name=d) for d in dir_nms]
        # mic(paths)
        d_df = parse_tensorboards(paths)
        # mic(d_df)

        y = 'loss'
        # y = 'ntp_acc'
        label = 'Next-Token-Prediction Accuracy'

        plot_tb(
            d_df, y=y, label=label, smooth_factor=dict(train=0.95, eval=0.5), steps_per_epoch=343,
            title='Reformer-base Training per-batch performance over Steps',
            # save=True
        )
    # check_plot_multiple()

    # def plot_trained_04_03():
    #     """
    #     Plot training for base reformer, trained for 256 epochs, until 128 epochs
    #     """
    #     dir_nms = ['2022-04-03_00-20-53', '2022-04-03_11-01-04']
    #     paths_ = [md_n_dir2tb_path(directory_name=d) for d in dir_nms]
    #     df = parse_tensorboards(paths_)
    #     steps_per_epoch = 29
    #     df = df[df.step < 128 * steps_per_epoch]
    #     # doesn't look nice
    #     # plot_tb(df, y=['loss', 'ntp_acc'], label=['Loss', 'Next-Token Prediction Accuracy'], save=False)
    #     plot_tb(
    #         df, y='loss', label='Loss',
    #         # title='Reformer-base Training per-batch performance over Steps',
    #         title='None',
    #         smooth_factor=0.95, figure_kwargs=dict(figsize=(9, 4)),
    #         save=True
    #     )
    # plot_trained_04_03()

    def plot_train_for_presentation():
        d_df = parse_tensorboard(md_n_dir2tb_path(directory_name='2022-04-11_00-26-05'))
        mic(d_df)

        od_fg = hex2rgb('#B1B8C5', normalize=True)
        od_bg = hex2rgb('#282C34', normalize=True)
        od_blue = hex2rgb('#619AEF', normalize=True)
        od_purple = hex2rgb('#C678DD', normalize=True)
        plt.style.use('dark_background')
        plt.rcParams.update({
            'axes.facecolor': od_bg, 'figure.facecolor': od_bg, 'savefig.facecolor': od_bg,
            'xtick.color': od_fg, 'ytick.color': od_fg, 'axes.labelcolor': od_fg,
            'grid.linewidth': 0.5, 'grid.alpha': 0.5,
            'axes.linewidth': 0.5,
        })

        plot_tb(
            d_df, y='loss',
            # title='Reformer-base Training per-batch performance over Steps',
            title='None',
            smooth_factor=dict(train=0.95, eval=0.5), figure_kwargs=dict(figsize=(9, 4)),
            save=True,
            cs=[od_blue, od_purple],
            steps_per_epoch=343
        )
    # plot_train_for_presentation()

    def plot_comparison_for_report():
        dir_nms_base = ['2022-04-15_13-42-56', '2022-04-15_18-45-49', '2022-04-16_16-08-03']
        dir_nms_aug = ['2022-04-17_22-53-41', '2022-04-18_08-56-05', '2022-04-19_13-48-54']
        paths_base = [md_n_dir2tb_path(directory_name=d) for d in dir_nms_base]
        paths_aug = [md_n_dir2tb_path(directory_name=d) for d in dir_nms_aug]

        y = 'ntp_acc'
        plt.figure(figsize=(9, 4))
        ax = plt.gca()
        s_fact = dict(train=0.95, eval=0.5)
        args = dict(
            smooth_factor=s_fact, ax=ax, show=False, smoothed_only=False, prefix_split=False,
            splits='eval', title='none'
        )

        od_blue = hex2rgb('#619AEF', normalize=True)
        od_purple = hex2rgb('#C678DD', normalize=True)
        df_base, df_aug = parse_tensorboards(paths_base), parse_tensorboards(paths_aug)
        plot_tb(df_base, y=y, ylabel='', plot_label='Reformer-base, vanilla', cs=[od_blue], **args)
        plot_tb(df_aug, y=y, ylabel='', plot_label='Reformer-base, key augmentation', cs=[od_purple], **args)
        plt.xlabel('epoch')
        plt.ylabel('Next-Token Prediction Accuracy (%)')

        title = 'Validation Next-Token Prediction Accuracy over Epochs'
        save_fig(title)
        plt.show()
    plot_comparison_for_report()
