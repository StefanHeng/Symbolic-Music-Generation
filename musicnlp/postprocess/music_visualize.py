from copy import deepcopy
from fractions import Fraction
from collections import Counter

from pandas.api.types import CategoricalDtype

from musicnlp.util import *
from musicnlp.model import LMTTokenizer
from musicnlp.preprocess import WarnLog
from musicnlp.postprocess import MusicStats


class MusicVisualize:
    """
    Visualize dataset info given json as extracted input representation

    See `preprocess.music_export.py`
    """
    def __init__(self, fl_nm):
        with open(os.path.join(fl_nm)) as f:
            self.dset: Dict = json.load(f)

        self.prec = self.dset['precision']
        assert self.prec >= 2
        self.tokenizer = LMTTokenizer(prec=self.prec)
        self.vocab: MusicVocabulary = self.tokenizer.vocab
        self.stats = MusicStats(prec=self.prec)
        self.df = None

    def _load_df(self):
        if self.df is None:
            self.df = self._get_song_info()

    def _get_song_info(self):
        entries: List[Dict] = self.dset['music']

        def extract_info(d: Dict):
            d = deepcopy(d)
            toks = self.tokenizer.tokenize(d.pop('text'))
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
        return pd.DataFrame([extract_info(e) for e in entries])

    def hist_wrapper(self, col_name: str, title: str, xlabel: str, callback: Callable = None, **kwargs):
        self._load_df()
        kwargs = dict(kde=True) | kwargs
        ax = sns.histplot(data=self.df, x=col_name, **kwargs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('count')
        if callback is not None:
            callback(ax)
        plt.show()

    def token_length_dist(self):
        self.hist_wrapper(col_name='n_token', title='Histogram of #encoded tokens per song', xlabel='#encoded tokens')

    def bar_count_dist(self):
        self.hist_wrapper(col_name='n_bar', title='Histogram of #bars per song', xlabel='#bars')

    def tuplet_count_dist(self):
        self.hist_wrapper(col_name='n_tup', title='Histogram of #tuplets per song', xlabel='#tuplets')

    def song_duration_dist(self):
        def callback(ax):
            x_tick_vals = plt.xticks()[0]
            ax.set_xticks(x_tick_vals, labels=[sec2mmss(v) for v in x_tick_vals])
        self._load_df()
        self.hist_wrapper(
            col_name='duration', title='Histogram of song duration', xlabel='duration (mm:ss)', callback=callback
        )

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
        self._load_df()
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
        self._load_df()
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

    def warning_type_dist(self, average=True):
        df = self.warn_info()
        cat = CategoricalDtype(categories=WarnLog.TYPES, ordered=True)
        assert not df.type.isnull().values.any()
        df.type = df.type.astype(cat, copy=False)
        ax = sns.barplot(data=df, y='type', x='average_count' if average else 'total_count')
        ax.set_xscale('log')
        plt.title('Bar plot of warning type, ordered by severity, across all songs')
        plt.ylabel('Warning type ' + 'per song' if average else 'in total')
        plt.xlabel('count')
        plt.show()


if __name__ == '__main__':
    from icecream import ic

    fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-03-01 02-29-29.json'
    fnm = os.path.join(config('path-export'), fnm)
    mv = MusicVisualize(fnm)

    def check_warn():
        df = mv.warn_info(as_counts=True)
        ic(df)
    # check_warn()

    def plots():
        # mv.token_length_dist()
        # mv.bar_count_dist()
        # mv.tuplet_count_dist()
        # mv.song_duration_dist()
        # mv.time_sig_dist()
        mv.tempo_dist()
        # mv.note_pitch_dist()
        # mv.note_duration_dist()
        # mv.warning_type_dist()
    plots()