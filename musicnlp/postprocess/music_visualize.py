import math
import json
import pickle
from copy import deepcopy
from typing import List, Dict, Iterable, Callable, Union
from fractions import Fraction
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from musicnlp.util import *
from musicnlp.vocab import COMMON_TEMPOS, COMMON_TIME_SIGS, MusicVocabulary, MusicTokenizer
from musicnlp.preprocess import WarnLog
from musicnlp.postprocess import MusicStats


def change_bar_width(ax, width: float = 0.5, orient: str = 'v'):
    """
    Modifies the bar width of a matplotlib bar plot

    Credit: https://stackoverflow.com/a/44542112/10732321
    """
    ca(orient=orient)
    is_vert = orient in ['v', 'vertical']
    for patch in ax.patches:
        current_width = patch.get_width() if is_vert else patch.get_height()
        diff = current_width - width
        patch.set_width(width) if is_vert else patch.set_height(width)
        patch.set_x(patch.get_x() + diff * .5) if is_vert else patch.set_y(patch.get_y() + diff * .5)


def barplot(
        data: pd.DataFrame = None,
        x: Union[Iterable[str], str] = None, y: Union[Iterable[float], str] = None,
        x_order: Iterable[str] = None,
        orient: str = 'v', with_value: bool = False, width: [float, bool] = 0.5,
        xlabel: str = None, ylabel: str = None, yscale: str = None, title: str = None, show: bool = True,
        ax=None, palette: str = 'husl', callback: Callable[[plt.Axes], None] = None, save: bool = False,
        **kwargs
):
    ca(orient=orient)
    if data is not None:
        df = data
        assert isinstance(x, str) and isinstance(y, str)
        df['x'], df['y'] = df[x], df[y]
    else:
        df = pd.DataFrame([dict(x=x_, y=y_) for x_, y_ in zip(x, y)])
        x_order = x
    if x_order:
        cat = CategoricalDtype(categories=x_order, ordered=True)  # Enforce ordering in plot
        df['x'] = df['x'].astype(cat, copy=False)
    is_vert = orient in ['v', 'vertical']
    x, y = ('x', 'y') if is_vert else ('y', 'x')
    if ax:
        kwargs['ax'] = ax
    if palette is not None:
        kwargs['palette'] = palette
    ax = sns.barplot(data=df, x=x, y=y, **kwargs)
    if with_value:
        ax.bar_label(ax.containers[0])
    if width:
        change_bar_width(ax, width, orient=orient)
    ax.set_xlabel(xlabel) if is_vert else ax.set_ylabel(xlabel)  # if None just clears the label
    ax.set_ylabel(ylabel) if is_vert else ax.set_xlabel(ylabel)
    if yscale:
        ax.set_yscale(yscale)
    if title:
        ax.set_title(title)
    if callback:
        callback(ax)
    if save:
        save_fig(title)
    if show:
        plt.show()
    return ax


class MusicVisualize:
    """
    Visualize dataset info given json as extracted input representation

    See `preprocess.music_export.py`
    """
    key_dnm = 'dataset_name'

    def __init__(
            self, filename: Union[str, List[str]], dataset_name: Union[str, List[str]] = None,
            color_palette: str = 'husl', hue_by_dataset: bool = True, cache: str = None,
    ):
        """
        :param filename: Path to a json dataset, or a list of paths, in which case datasets are concatenated
            See `preprocess.music_export.py`
        :param dataset_name: Datasets names, if given, should correspond to filenames
        :param hue_by_dataset: If true, automatically color-code statistics by dataset name
        """
        def _load_single(f_: str, dnm: str = None) -> Dict:
            with open(f_, 'r') as f:
                ds = json.load(f)
            ds['music'] = ds['music'][:256]  # TODO: debugging
            if dnm:
                for s in ds['music']:
                    s['dataset_name'] = dnm
            return ds

        def get_prec(ds: Dict) -> int:
            return get(ds, 'extractor_meta.precision')

        self.dset: Dict
        if isinstance(filename, str):
            if dataset_name:
                assert isinstance(dataset_name, str), \
                    f'Dataset name given should be a string for single filename, ' \
                    f'but got {logi(dataset_name)} with type {logi(type(dataset_name))}'
            self.dset = _load_single(filename, dataset_name)
        else:
            if dataset_name:
                assert isinstance(dataset_name, list), \
                    f'Dataset name given should be a list for multiple filenames, ' \
                    f'but got {logi(dataset_name)} with type {logi(type(dataset_name))}'
            else:
                dataset_name = [None] * len(filename)
            dset = [_load_single(f, dnm) for f, dnm in zip(filename, dataset_name)]
            assert all(ds['extractor_meta'] == dset[0]['extractor_meta'] for ds in dset)
            self.dset = dict(
                music=sum([d['music'] for d in dset], []),
                extractor_meta=dset[0]['extractor_meta']
            )

        self.prec = get_prec(self.dset)
        assert self.prec >= 2
        self.tokenizer = MusicTokenizer(precision=self.prec)
        self.vocab: MusicVocabulary = self.tokenizer.vocab
        self.stats = MusicStats(prec=self.prec)
        self._df = None
        self.color_palette = color_palette
        if hue_by_dataset:
            assert dataset_name is not None, f'{logi("dataset_name")} is required for color coding'
        self.hue_by_dataset = hue_by_dataset
        self.cache = cache

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            if self.cache:
                if os.path.exists(self.cache):
                    with open(self.cache, 'rb') as f:
                        self._df = pickle.load(f)
                else:
                    self._df = self._get_song_info()
                    with open('visualize_df.pkl', 'wb') as f:
                        pickle.dump(self._df, f)
            else:
                self._df = self._get_song_info()
        return self._df

    def _get_song_info(self):
        entries: List[Dict] = self.dset['music']

        def extract_info(d: Dict):
            d = deepcopy(d)
            toks = self.tokenizer.tokenize(d.pop('score'))
            d['n_token'] = len(toks)
            counter_toks = Counter(toks)
            d['n_bar'] = counter_toks[self.vocab.start_of_bar]
            d['n_tup'] = counter_toks[self.vocab.start_of_tuplet]
            del d['warnings']
            ttc = self.stats.vocab_type_counts(toks)
            # Only 1 per song
            # (numer, denom), d['tempo'] = list(ttc['time_sig'].keys())[0], list(ttc['tempo'].keys())[0]
            # d['time_sig'] = f'{numer}/{denom}'
            d['time_sig'], d['tempo'] = list(ttc['time_sig'].keys())[0], list(ttc['tempo'].keys())[0]
            d['duration_count'], d['pitch_count'] = ttc['duration'], ttc['pitch']
            d['weighted_pitch_count'] = self.stats.weighted_pitch_counts(toks)
            return d
        ds = []
        for e in tqdm(entries, desc='Extracting song info', unit='song'):
            ds.append(extract_info(e))
        return pd.DataFrame(ds)

    def hist_wrapper(
            self, data: pd.DataFrame = None, col_name: str = None, title: str = None, xlabel: str = None,
            ylabel: str = None, yscale: str = None,
            callback: Callable = None, show=True, save: bool = False,
            upper_percentile: float = None,
            **kwargs
    ):
        kwargs = dict(kde=True) | kwargs
        if self.hue_by_dataset:
            kwargs['hue'] = 'dataset_name'
        data = data if data is not None else self.df
        ax = sns.histplot(data=data, x=col_name, palette=self.color_palette, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or 'count')
        if title is not None:
            plt.title(title)
        if upper_percentile:
            vs = self.df[col_name]
            q = upper_percentile if isinstance(upper_percentile, float) else 99.7  # ~3std
            mi, ma = vs.min(), np.percentile(vs, q=q)
            ax.set_xlim([mi, ma])
        if yscale:
            plt.yscale(yscale)
        if callback is not None:
            callback(ax)
        if show:
            plt.show()
        if save:
            save_fig(title)
        return ax

    def token_length_dist(self, **kwargs):
        args = dict(col_name='n_token', title='Histogram of token length', xlabel='#token')
        if kwargs is not None:
            args.update(kwargs)
        self.hist_wrapper(**args)

    def bar_count_dist(self):
        self.hist_wrapper(col_name='n_bar', title='Histogram of #bars per song', xlabel='#bar')

    def tuplet_count_dist(self):
        title = 'Histogram of #tuplets per song'
        self.hist_wrapper(col_name='n_tup', title=title, xlabel='#tuplet', upper_percentile=True)

    def song_duration_dist(self, **kwargs):
        def callback(ax):
            x_tick_vals = [v for v in plt.xticks()[0] if v >= 0]
            ax.set_xticks(x_tick_vals, labels=[sec2mmss(v) for v in x_tick_vals])
        args = dict(
            col_name='duration', title='Histogram of song duration', xlabel='duration (mm:ss)', callback=callback
        )
        if kwargs is not None:
            args.update(kwargs)
        self.hist_wrapper(**args)

    def time_sig_dist(self, kind: str = 'bar', **kwargs):
        title = 'Distribution of Time Signature'
        if kind == 'hist':
            def callback(ax):
                ax.set_yscale('log')
            cnm = 'time_sig'
            self.hist_wrapper(
                col_name=cnm, title=title, xlabel='Time Signature', yscale='log', kde=False, callback=callback, **kwargs
            )
        else:
            def callback(ax):
                plt.gcf().canvas.draw()  # so that labels are rendered
                xtick_lbs = ax.get_xticklabels()
                c_bad = hex2rgb('#E06C75', normalize=True)
                com_tss = [f'{ts[0]}/{ts[1]}' for ts in COMMON_TIME_SIGS]
                for t in xtick_lbs:  # TODO: fix, with `plt.show()` the colors don't show up, but `savefig()` works
                    if t.get_text() not in com_tss:
                        t.set_color(c_bad)
            counter = Counter(self.df.time_sig)
            # sort by duration of the time signature
            tss_uncom = sorted([ts for ts in counter.keys() if ts not in COMMON_TIME_SIGS], key=lambda ts: ts[0]/ts[1])
            tss = COMMON_TIME_SIGS + tss_uncom
            tss_print = [f'{numer}/{denom}' for numer, denom in tss]
            df = self._count_by_dataset('time_sig')
            df['time_sig'] = df['time_sig'].map(lambda ts: f'{ts[0]}/{ts[1]}')  # so that the category ordering works
            args = dict(
                data=df, x='time_sig', y='count',
                hue=self.key_dnm, x_order=tss_print,
                xlabel='Time Signature', ylabel='count', title=title, yscale='log', width=False, callback=callback
            )
            args |= kwargs
            barplot(**args)

    def tempo_dist(self):
        def callback(ax):
            plt.gcf().canvas.draw()  # so that labels are rendered
            xtick_lbs = ax.get_xticklabels()
            c_bad = hex2rgb('#E06C75', normalize=True)
            for t in xtick_lbs:
                if int(t.get_text()) not in COMMON_TEMPOS:
                    t.set_color(c_bad)
        title, xlab = 'Histogram of tempo per song', 'Tempo (bpm)'
        return self.hist_wrapper(col_name='tempo', title=title, xlabel=xlab, yscale='log', kde=False, callback=callback)

    def _count_column(self, col_name: str) -> Dict[str, Counter]:
        dnm2counts = defaultdict(Counter)
        for dnm in self.df[self.key_dnm].unique():
            df = self.df[self.df[self.key_dnm] == dnm]
            # dnm2counts[dnm].update(df[col_name].value_counts())
            for d in df[col_name]:  # TODO: optimize
                dnm2counts[dnm].update((d,))
        return dnm2counts

    def _count_by_dataset(self, col_name: str) -> pd.DataFrame:
        dfs, cols = [], [col_name, 'count', self.key_dnm]
        for dnm, counts in self._count_column(col_name).items():
            dfs.append(pd.DataFrame([(k, v, dnm) for k, v in counts.items()], columns=cols))
        return pd.concat(dfs, ignore_index=True)

    def note_pitch_dist(self, weighted=True):
        dnm2counts = defaultdict(Counter)
        k_dnm = 'dataset_name'
        for dnm in self.df[k_dnm].unique():
            df = self.df[self.df[k_dnm] == dnm]
            for d in (df.weighted_pitch_count if weighted else df.pitch_count):
                dnm2counts[dnm].update(d)
        dfs = []
        for dnm, counts in dnm2counts.items():
            dfs.append(pd.DataFrame([(k, v, dnm) for k, v in counts.items()], columns=['pitch', 'count', k_dnm]))
        df = pd.concat(dfs, ignore_index=True)
        ma, mi = df.pitch.max(), df.pitch.min()
        assert mi == self.vocab.compact(self.vocab.rest)

        def callback(ax):
            plt.gcf().canvas.draw()
            pch_ints = [-1, *range(6, ma + 6, 6)]
            ax.set_xticks(pch_ints, labels=[self.tokenizer.vocab.pitch_midi2name(p) for p in pch_ints])

        title, xlab = 'Histogram of pitch across all songs', 'Pitch'
        ylab = 'weighted count' if weighted else 'count'
        return self.hist_wrapper(
            data=df, col_name='pitch', weights='count',
            title=title, xlabel=xlab, ylabel=ylab, kde=True, callback=callback,
            bins=ma-mi+1, kde_kws=dict(bw_adjust=0.5)
            # , stat='density'
        )

    def note_duration_dist(self):
        """
        Tuplet notes contribute to a single duration, i.e. all quantized durations
        """
        counts = Counter()
        for d in self.df.duration_count:
            counts.update(d)
        # df = pd.DataFrame([(k, v) for k, v in counts.items()], columns=['duration', 'count'])
        # bound = math.ceil(df.duration.max())
        # ic(max(counts.keys()), counts.keys())
        bound = min(max(counts.keys()), 6)  # Hard coded for now, based on common time signature

        # cat = CategoricalDtype(categories=self.vocab.get_durations(bound=bound, exp='dur'), ordered=True)
        # Number of colors needed for an integer group
        # e.g. precision = 5, smallest duration 1/8, needs 4 colors, for [1, 1/2, 1/4, 1/8]
        n_color_1 = (self.prec-2) + 1
        n_color = n_color_1 * bound
        # df.duration = df.duration.astype(cat, copy=False)

        durs_above = sorted([d for d in counts.keys() if d > bound])
        durs = self.vocab.get_durations(bound=bound, exp='dur')
        durs_str = durs + durs_above
        cs = sns.color_palette(palette='husl', n_colors=n_color + len(durs_above))

        # Colors are locally ordered by precision, added with shades based on global magnitude
        def get_category_idx(c: Union[int, Fraction]):
            if isinstance(c, int):
                idx_local, group = 0, c-1
            else:
                idx_local = math.log2(c.denominator)
                assert idx_local.is_integer()
                idx_local, group = int(idx_local), math.floor(c)
            return idx_local + n_color_1 * group
        # cs = [cs[get_category_idx(c)] for c in df.duration.cat.categories]
        # ic(durs, durs_str, durs_above)
        cs = [cs[get_category_idx(c)] for c in durs] + [cs[-1] for c in durs_above]

        # ic(cs)

        def callback(ax):
            ax.set_yscale('log')

        title, xlab = 'Bar plot of duration across all songs', 'Duration (quarter length)'
        barplot(x=durs_str, y=[counts[d] for d in durs_str], palette=cs,
                xlabel=xlab, ylabel='count', title=title, callback=callback
                )

    @property
    def n_song(self) -> int:
        return len(self.dset['music'])

    def warn_info(self, as_counts=True) -> pd.DataFrame:
        """
        Aggregate warnings as a pandas Dataframe

        :param as_counts: If true, get counts about warnings logged during extraction
            returns warning counts per song
        """
        entries = self.dset['music']

        def entry2df(d: Dict) -> pd.DataFrame:
            def prep_warn(d_warn: Dict) -> Dict:
                d_out = dict()
                d_out['src'] = d['title']
                d_out['type'] = d_warn.pop('warn_name', None)
                d_out['args'] = json.dumps(d_warn)
                return d_out
            return pd.DataFrame([prep_warn(d) for d in d['warnings']])
        df = pd.concat([entry2df(e) for e in entries])
        if as_counts:
            counts = df.type.value_counts()
            df = counts.to_frame(name='total_count').reset_index()  # Have `index` as a column
            df.rename(columns={'index': 'type'}, inplace=True)
            df['average_count'] = df.apply(lambda x: x.total_count/self.n_song, axis=1)
        return df

    def warning_type_dist(self, average=True, title: str = None, show=True, bar_kwargs: Dict = None):
        df = self.warn_info()
        cat = CategoricalDtype(categories=WarnLog.TYPES, ordered=True)
        assert not df.type.isnull().values.any()
        df.type = df.type.astype(cat, copy=False)
        if bar_kwargs is None:
            bar_kwargs = dict()
        ax = sns.barplot(data=df, y='type', x='average_count' if average else 'total_count', **bar_kwargs)
        ax.set_xscale('log')
        plt.ylabel('Warning type')
        typ = 'per song' if average else 'in total'
        plt.xlabel(f'count {typ}')
        if title is None:
            title = 'Bar plot of warning type, ordered by severity, across all songs'
        if title != 'None':
            plt.title(title)
        if show:
            plt.show()


if __name__ == '__main__':
    import os

    from icecream import ic

    import musicnlp.util.music as music_util

    fnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-51-01.json'
    fnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
              'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-10_19-49-52.json'
    fnms = [os.path.join(music_util.get_processed_path(), dnm) for dnm in [fnm_909, fnm_lmd]]
    mv = MusicVisualize(filename=fnms, dataset_name=['POP909', 'LCS'], hue_by_dataset=True)  # for `LMD-cleaned-subset`

    def check_warn():
        df = mv.warn_info(as_counts=True)
        ic(df)
    # check_warn()

    def check_uncommon_tempos():
        df = mv.df
        tempos = df.tempo.unique()
        ic(tempos)
        ic(set(tempos) - set(COMMON_TEMPOS))
    # check_uncommon_tempos()

    def plots():
        # mv.token_length_dist(hue='dataset_name')
        # mv.token_length_dist()
        # mv.bar_count_dist()
        # mv.tuplet_count_dist()
        # mv.song_duration_dist()
        mv.time_sig_dist()
        # mv.tempo_dist()
        # ic(mv.df)
        # mv.note_pitch_dist()
        # mv.note_duration_dist()
        # mv.warning_type_dist()
    plots()

    fig_sz = (9, 5)

    def save_plots_for_report():
        plt.figure(figsize=fig_sz)
        ax = plt.gca()
        args = dict(new_figure=True, ax=ax, title=None)

        def token_len():
            mv.token_length_dist(**args)
            return 'Distribution of token length for each song'
        # title = token_len()

        def song_duration():
            mv.song_duration_dist(**args)
            return 'Distribution of song duration'
        # plt.show()
        title = song_duration()
        save_fig(f'{title}, {now(for_path=True)}')
    # save_plots_for_report()

    def save_for_report_warn():
        # plt.figure(figsize=(9, 5))
        mv.warning_type_dist(title='None', show=False, bar_kwargs=dict(figure=plt.figure(figsize=fig_sz)))
        title = 'Average #Warnings for each song'
        save_fig(f'{title}, {now(for_path=True)}')
    # save_for_report_warn()

    fig_sz = (9, 9)

    def plots_for_presentation():
        # Colors from One Dark theme
        od_fg = hex2rgb('#B1B8C5', normalize=True)
        od_bg = hex2rgb('#282C34', normalize=True)
        od_blue = hex2rgb('#619AEF', normalize=True)
        od_purple = hex2rgb('#C678DD', normalize=True)

        plt.style.use('dark_background')
        sns.set(style='ticks', context='talk')
        plt.rcParams.update({
            'axes.facecolor': od_bg, 'figure.facecolor': od_bg, 'savefig.facecolor': od_bg,
            'xtick.color': od_fg, 'ytick.color': od_fg, 'axes.labelcolor': od_fg,
            'grid.linewidth': 0.5, 'grid.alpha': 0.5,
            'axes.linewidth': 0.5,
        })

        plt.figure(figsize=fig_sz)
        ax = plt.gca()
        args = dict(new_figure=False, ax=ax, title=None)

        def token_len():
            args['color'] = od_blue
            mv.token_length_dist(**args)
            return mv.df.n_token, 'Distribution of token length for each song'

        def song_duration():
            args.update(dict(color=od_purple, upper_percentile=97))
            mv.song_duration_dist(**args)
            return mv.df.duration, 'Distribution of song duration'

        vals, title = song_duration()
        save_fig(f'{title}, {now(for_path=True)}')
    # plots_for_presentation()

    def plot_for_report():
        mv.time_sig_dist(title=None, show=False)  # TODO: with bar plot instead?
        title = 'Distribution of Time Signature'
        save_fig(title)
    # plot_for_report()

