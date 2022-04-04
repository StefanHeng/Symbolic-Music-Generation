from typing import Sequence

import tensorflow as tf
from tensorflow.core.util import event_pb2

from musicnlp.util import *


def parse_tensorboard(path) -> pd.DataFrame:
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

    fnms = list(glob.iglob(os.path.join(path, '**/events.out.tfevents*'), recursive=True))
    assert len(fnms) == 1, f'Expect one events.out.tfevents file, found {logi(len(fnms))}'
    fnm = fnms[0]
    events = [event_pb2.Event.FromString(rec.numpy()) for rec in tf.data.TFRecordDataset(fnm)]
    events = [parse_single(e) for e in events if len(e.summary.value)]
    events.sort(key=lambda e: (e['step'], e['wall_time']))
    events = [list(v) for k, v in itertools.groupby(events, key=lambda e: e['step'])]

    pattern_name = re.compile(r'(?P<tag>.*)/(?P<key>.*)')

    def name2tag_n_key(name: str) -> Tuple[str, str]:
        m = pattern_name.match(name)
        return m.group('tag'), m.group('key')

    def group_single(group_events: List[Dict]):  # expects certain formatting of the `name`
        d_out = dict(step=group_events[0]['step'])  # pick one arbitrarily
        # keep the key, discard the tag for now
        return d_out | {name2tag_n_key(e['name'])[1]: e['value'] for e in group_events}
    events = [group_single(e) for e in events]
    df_ = pd.DataFrame(events)
    mi, ma = df_.step.min(), df_.step.max()
    assert np.array_equal(df_.step.to_numpy(), np.arange(mi, ma + 1)), \
        f'Expect step to be continuously increasing integer range'
    return df_


def parse_tensorboards(paths: List[str]) -> pd.DataFrame:
    """
    Parse multiple tensorboard files & merge them by steps
        Intended for resuming training from checkpoints
    :param paths: Paths to tensorboard files, in increasing order of ste[s
    """
    dfs = [parse_tensorboard(path) for path in paths]
    mis, mas = [df.step.min() for df in dfs], [df.step.max() for df in dfs]
    mis_correct = all(mi < mis[i+1] for i, mi in enumerate(mis[:-1]))
    mas_correct = all(ma < mas[i+1] for i, ma in enumerate(mas[:-1]))
    assert mis_correct and mas_correct, 'Steps for each consecutive tensorboard should be increasing'
    dfs_ = [dfs[-1]]  # add dataframes backwards, earlier overlapping data will be truncated
    for i, df_ in enumerate(reversed(dfs[:-1])):
        dfs_.append(df_[df_.step < mis[i+1]])
    return pd.concat(reversed(dfs_)).reset_index(drop=True)


def smooth(vals: Sequence[float], factor: float) -> np.array:
    last = vals[0]
    smoothed = np.empty(len(vals), dtype=np.float32)
    for i, v in enumerate(vals):
        last = smoothed[i] = factor*last + (1-factor) * v
    return smoothed


def plot_tb(
        df: pd.DataFrame, y: Union[str, List[str]] = 'loss', save=False,
        label: Union[str, List[str]] = None, smooth_factor=0.9, figure_kwargs: Dict = None,
        title: str = None,
        smaller_plot: bool = False
):
    if isinstance(y, (list, tuple)):
        assert len(y) == 2, 'Only 2 values supported for multiple-value plot'
    else:
        y = [y]
    label = label if label is not None else y
    if not isinstance(label, (list, tuple)):
        label = [label]
    if figure_kwargs is None:
        figure_kwargs = dict()
    plt.figure(**figure_kwargs)
    x = df.step
    cs = sns.color_palette(palette='husl', n_colors=7)

    def plot_single(idx, y_, ax=plt.gca(), c=None):
        y__ = df[y_]
        if 'acc' in y_:
            y__ *= 100
        y_s = smooth(y__, factor=smooth_factor)
        if c is None:
            c = cs[idx]
        if smaller_plot:  # TODO: kinda ugly & hard-coded
            args_ori = LN_KWARGS | dict(ls='None', c=c, alpha=0.5, ms=0.2)
            args_smooth = LN_KWARGS | dict(c=cs[-2], lw=0.75, marker=None)
        else:
            args_ori = LN_KWARGS | dict(ls=':', c=c, alpha=0.7)
            args_smooth = LN_KWARGS | dict(c=c, lw=0.75)
        ax.plot(x, y__, **args_ori)
        # ic(label[idx])
        return ax.plot(x, y_s, **args_smooth, label=label[idx])
    plt.xlabel('Step')
    if len(y) == 2:
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        l1 = plot_single(0, y[0], ax=ax1)
        l2 = plot_single(1, y[1], ax=ax2)
        ax1.set_ylabel(label[0])
        ax2.set_ylabel(label[1])
        plt.legend(handles=l1+l2)
    else:
        plot_single(0, y[0], c=cs[-3])
        plt.ylabel(label[0])
        plt.legend()
    save_title = 'Training per-batch performance over Steps'
    if title is None:
        title = save_title
    if title != 'None':
        plt.title(title)
        save_title = title
    if save:
        save_fig(f'{save_title}, {now(for_path=True)}')
    else:
        plt.show()


def md_n_dir2tb_path(model_name: str = 'reformer', directory_name: str = None) -> str:
    return os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, model_name, directory_name)


if __name__ == '__main__':
    from icecream import ic
    # path_ = '/Users/stefanh/Documents/UMich/Research/Music with NLP/Symbolic-Music-Generation/models'
    # directory_name = '2022-04-01_09-40-48'

    def check_plot_single():
        df = parse_tensorboard(md_n_dir2tb_path(directory_name='2022-04-03_00-20-53'))
        ic(df)
        plot_tb(df, y='ntp_acc', save=False)
    # check_plot_single()

    def check_plot_multiple():
        dir_nms = ['2022-04-03_00-20-53', '2022-04-03_11-01-04']
        paths_ = [md_n_dir2tb_path(directory_name=d) for d in dir_nms]
        df = parse_tensorboards(paths_)
        ic(df)
    # check_plot_multiple()

    def plot_trained_04_03():
        """
        Plot training for base reformer, trained for 256 epochs, until 128 epochs
        """
        dir_nms = ['2022-04-03_00-20-53', '2022-04-03_11-01-04']
        paths_ = [md_n_dir2tb_path(directory_name=d) for d in dir_nms]
        df = parse_tensorboards(paths_)
        steps_per_epoch = 29
        df = df[df.step < 128 * steps_per_epoch]
        # doesn't look nice
        # plot_tb(df, y=['loss', 'ntp_acc'], label=['Loss', 'Next-Token Prediction Accuracy'], save=False)
        plot_tb(
            df, y='loss', label='Loss',
            # title='Reformer-base Training per-batch performance over Steps',
            title='None',
            smooth_factor=0.95, smaller_plot=True, figure_kwargs=dict(figsize=(9, 4)),
            save=True
        )
    plot_trained_04_03()

