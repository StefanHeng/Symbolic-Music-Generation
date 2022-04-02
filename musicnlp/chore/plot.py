from tensorflow.python.summary.summary_iterator import summary_iterator

from musicnlp.util import *


def parse_tensorboard(path) -> pd.DataFrame:
    """
    Modified from https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/

    parse a tensor board, only 1 tag supported, each time step should have the same fixed number of values
    """
    def parse_single(tfevent):
        assert len(tfevent.summary.value) == 1
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    fnms = list(glob.iglob(os.path.join(path, '**/events.out.tfevents*'), recursive=True))
    assert len(fnms) == 1, f'Expect one events.out.tfevents file, found {len(fnms)}'
    fnm = fnms[0]

    events = [parse_single(e) for e in summary_iterator(fnm) if len(e.summary.value)]
    events.sort(key=lambda e: (e['step'], e['wall_time']))
    events = [list(v) for k, v in itertools.groupby(events, key=lambda e: e['step'])]
    # ic(events[:20])

    pattern_name = re.compile(r'(?P<tag>.*)/(?P<key>.*)')
    # pattern_name = re.compile(r'^(.*)/(.*)$')

    def name2tag_n_key(name: str) -> Tuple[str, str]:
        m = pattern_name.match(name)
        # ic(name, m)
        return m.group('tag'), m.group('key')

    def group_single(group_events: List[Dict]):  # expects certain formatting of the `name`
        d_out = dict(step=group_events[0]['step'])  # pick one arbitrarily
        # keep the key, discard the tag for now
        return d_out | {name2tag_n_key(e['name'])[1]: e['value'] for e in group_events}
        # name, key = name2tag_n_key(group_events[0]['name'])
        # ic(group_events)
        # exit(1)
    # ic(events[:20])
    events = [group_single(e) for e in events]
    df = pd.DataFrame(events)
    mi, ma = df.step.min(), df.step.max()
    assert np.array_equal(df.step.to_numpy(), np.arange(mi, ma + 1)), \
        f'Expect step to be continuously increasing integer range'
    return df


def plot_tb_loss(df: pd.DataFrame, save=False):
    fig = plt.figure()
    plt.plot(df.step, df.loss, **LN_KWARGS, label='Reformer-small')
    title = 'Reformer Training Loss'
    plt.title(title)
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Loss')
    if save:
        save_fig(fig, f'{title}, {now(for_path=True)}')
    else:
        plt.show()


if __name__ == '__main__':
    from icecream import ic
    # path_ = '/Users/stefanh/Documents/UMich/Research/Music with NLP/Symbolic-Music-Generation/models'
    model_name, directory_name = 'reformer', '2022-04-01_09-40-48'
    path_ = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, model_name, directory_name)
    df_ = parse_tensorboard(path_)

    ic(df_)
    plot_tb_loss(df_)
