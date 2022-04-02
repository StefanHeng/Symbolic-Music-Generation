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
    assert len(fnms) == 1, f'Expect one events.out.tfevents file, found {len(fnms)}'
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
    df = pd.DataFrame(events)
    mi, ma = df.step.min(), df.step.max()
    assert np.array_equal(df.step.to_numpy(), np.arange(mi, ma + 1)), \
        f'Expect step to be continuously increasing integer range'
    return df


def smooth(vals: Sequence[float], factor: float) -> np.array:
    last = vals[0]
    smoothed = np.empty(len(vals), dtype=np.float32)
    for i, v in enumerate(vals):
        last = smoothed[i] = v_ = factor*last + (1-factor) * v
    return smoothed


def plot_tb_loss(df: pd.DataFrame, save=False, smooth_factor=0.9):
    fig = plt.figure()
    x, y = df.step, df.loss
    c = sns.color_palette(palette='husl', n_colors=7)[-2]
    y_s = smooth(y, factor=smooth_factor)
    args_ori = LN_KWARGS | dict(ls=':', c=c, alpha=0.7)
    args_smooth = LN_KWARGS | dict(c=c, lw=0.75)
    plt.plot(x, y, **args_ori)
    plt.plot(x, y_s, **args_smooth, label='Reformer-small')
    title = 'Reformer Training Loss'
    plt.title(title)
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Loss')
    if save:
        save_fig(f'{title}, {now(for_path=True)}')
    else:
        plt.show()


if __name__ == '__main__':
    from icecream import ic
    # path_ = '/Users/stefanh/Documents/UMich/Research/Music with NLP/Symbolic-Music-Generation/models'
    model_name, directory_name = 'reformer', '2022-04-01_09-40-48'
    path_ = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, model_name, directory_name)
    df_ = parse_tensorboard(path_)

    ic(df_)
    plot_tb_loss(df_, save=True)
