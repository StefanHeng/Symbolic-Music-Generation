from copy import deepcopy
from fractions import Fraction
from collections import Counter

from pandas.api.types import CategoricalDtype

from musicnlp.util import *
from musicnlp.vocab import COMMON_TEMPOS, MusicVocabulary, MusicTokenizer
from musicnlp.preprocess import WarnLog
from musicnlp.postprocess import MusicStats


class MusicVisualize:
    """
    Visualize dataset info given json as extracted input representation

    See `preprocess.music_export.py`
    """
    def __init__(self, filenames: Union[str, List[str]]):
        """
        :param filenames: Path to a json dataset, or a list of paths, in which case datasets are concatenated
            See `preprocess.music_export.py`
        """
        def _load_single(f_: str) -> Dict:
            with open(f_, 'r') as f:
                return json.load(f)

        def get_prec(ds: Dict) -> int:
            return get(ds, 'extractor_meta.precision')

        self.dset: Dict
        if isinstance(filenames, str):
            self.dset = _load_single(filenames)
        else:
            dset = [_load_single(f) for f in filenames]
            assert all(ds['extractor_meta'] == dset[0]['extractor_meta'] for ds in dset)
            self.dset = dict(
                music=sum([d['music'] for d in dset], []),
                extractor_meta=dset[0]['extractor_meta']
            )

        self.prec = get_prec(self.dset)
        assert self.prec >= 2
        self.tokenizer = MusicTokenizer(prec=self.prec)
        self.vocab: MusicVocabulary = self.tokenizer.vocab
        self.stats = MusicStats(prec=self.prec)
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._get_song_info()
        return self._df

    def _get_song_info(self):
        entries: List[Dict] = self.dset['music']
        # entries = entries[:256]  # TODO: debugging

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
            (numer, denom), d['tempo'] = list(ttc['time_sig'].keys())[0], list(ttc['tempo'].keys())[0]
            d['time_sig'] = f'{numer}/{denom}'
            d['duration_count'], d['pitch_count'] = ttc['duration'], ttc['pitch']
            d['weighted_pitch_count'] = self.stats.weighted_pitch_counts(toks)
            return d
        ds = []
        for e in tqdm(entries, desc='Extracting song info', unit='song'):
            ds.append(extract_info(e))
        return pd.DataFrame(ds)

    def hist_wrapper(
            self, col_name: str, title: str, xlabel: str, callback: Callable = None, new_figure=True,
            upper_percentile: float = None,
            **kwargs
    ):
        kwargs = dict(kde=True) | kwargs
        if not new_figure:
            assert 'ax' in kwargs, f'If not {logi("new_figure")}, {logi("ax")} must be passed in'
        ax = sns.histplot(data=self.df, x=col_name, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel('count')
        if title is not None:
            plt.title(title)
        if upper_percentile:
            vs = self.df[col_name]
            q = upper_percentile if isinstance(upper_percentile, float) else 99.7  # ~3std
            mi, ma = vs.min(), np.percentile(vs, q=q)
            ax.set_xlim([mi, ma])
        if callback is not None:
            callback(ax)
        if new_figure:
            plt.show()

    def token_length_dist(self, **kwargs):
        args = dict(col_name='n_token', title='Histogram of token length', xlabel='#token')
        if kwargs is not None:
            args.update(kwargs)
        self.hist_wrapper(**args)

    def bar_count_dist(self):
        self.hist_wrapper(col_name='n_bar', title='Histogram of #bars per song', xlabel='#bars')

    def tuplet_count_dist(self):
        self.hist_wrapper(col_name='n_tup', title='Histogram of #tuplets per song', xlabel='#tuplets')

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

    def time_sig_dist(self):
        self.hist_wrapper(
            col_name='time_sig', title='Histogram of time signature per song', xlabel='Time Signature', kde=False
        )

    def tempo_dist(self):
        def callback(ax):
            ax.set_yscale('log')
        self.hist_wrapper(
            col_name='tempo', title='Histogram of tempo per song', xlabel='Tempo/BPM',
            kde=False, callback=callback
        )

    def note_pitch_dist(self, weighted=True):
        counts = Counter()
        for d in (self.df.weighted_pitch_count if weighted else self.df.pitch_count):
            counts.update(d)
        df = pd.DataFrame([(k, v) for k, v in counts.items()], columns=['pitch', 'count'])
        ma, mi = df.pitch.max(), df.pitch.min()
        assert mi == self.vocab.compact(self.vocab.rest)
        ax = sns.histplot(
            data=df, x='pitch', weights='count', kde=True,
            bins=ma-mi+1, kde_kws=dict(bw_adjust=0.5)
        )
        pch_ints = [-1, *range(6, ma+6, 6)]
        ax.set_xticks(pch_ints, labels=[self.tokenizer.vocab.pitch_midi2name(p) for p in pch_ints])
        plt.title('Histogram of pitch across all songs')
        plt.xlabel('Pitch')
        plt.ylabel('count, weighted' if weighted else 'count')
        plt.show()

    def note_duration_dist(self):
        """
        Tuplet notes contribute to a single duration, i.e. all quantized durations
        """
        counts = Counter()
        for d in self.df.duration_count:
            counts.update(d)
        df = pd.DataFrame([(k, v) for k, v in counts.items()], columns=['duration', 'count'])
        bound = math.ceil(df.duration.max())
        cat = CategoricalDtype(categories=self.vocab.get_durations(bound=bound, exp='dur'), ordered=True)
        # Number of colors needed for an integer group
        # e.g. precision = 5, smallest duration 1/8, needs 4 colors, for [1, 1/2, 1/4, 1/8]
        n_color_1 = (self.prec-2) + 1
        n_color = n_color_1 * bound
        df.duration = df.duration.astype(cat, copy=False)
        cs = sns.color_palette(palette='husl', n_colors=n_color)

        # Colors are locally ordered by precision, added with shades based on global magnitude
        def get_category_idx(c: Union[int, Fraction]):
            if isinstance(c, int):
                idx_local, group = 0, c-1
            else:
                idx_local = math.log2(c.denominator)
                assert idx_local.is_integer()
                idx_local, group = int(idx_local), math.floor(c)
            return idx_local + n_color_1 * group
        cs = [cs[get_category_idx(c)] for c in df.duration.cat.categories]
        sns.barplot(data=df, x='duration', y='count', palette=cs)
        plt.title('Bar plot of duration across all songs')
        plt.xlabel('Duration (quarter length)')
        plt.ylabel('count')
        plt.show()

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
    from icecream import ic

    dnm_909 = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-10_12-51-01.json'
    dnm_lmd = 'musicnlp music extraction, dnm=LMD-cleaned-subset, ' \
              'n=10269, meta={mode=melody, prec=5, th=1}, 2022-04-10_19-49-52.json'
    dnms = [os.path.join(get_processed_path(), dnm) for dnm in [dnm_909, dnm_lmd]]
    mv = MusicVisualize(dnms)

    def check_warn():
        df = mv.warn_info(as_counts=True)
        ic(df)
    # check_warn()

    def check_uncommon_tempos():
        df = mv.load_df()
        tempos = df.tempo.unique()
        ic(tempos)
        ic(set(tempos) - set(COMMON_TEMPOS))
    # check_uncommon_tempos()

    def plots():
        mv.token_length_dist()
        # mv.bar_count_dist()
        # mv.tuplet_count_dist()
        # mv.song_duration_dist()
        # mv.time_sig_dist()
        # mv.tempo_dist()
        # ic(mv.df)
        # mv.note_pitch_dist()
        # mv.note_duration_dist()
        # mv.warning_type_dist()
    # plots()

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
        # ic(od_bg, od_blue)

        # sns.set_style('dark_background')
        # sns.set(style="ticks", context="talk")
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
    plots_for_presentation()
