from collections import Counter

from musicnlp.util import *
from musicnlp.model import LMTTokenizer
from musicnlp.postprocess import MusicStats

class MusicVisualize:
    """
    Visualize dataset info given json as extracted input representation

    See `preprocess.music_export.py`
    """
    def __init__(self, fl_nm):
        with open(os.path.join(fl_nm)) as f:
            self.dset: Dict = json.load(f)

        prec = self.dset['precision']
        self.tokenizer = LMTTokenizer(prec=prec)
        self.stats = MusicStats(prec=prec)
        self.df = self._get_song_info()

    def _get_song_info(self):
        entries: List[Dict] = self.dset['music']

        def extract_info(d: Dict):
            toks = self.tokenizer.tokenize(d.pop('text'))
            d['n_token'] = len(toks)
            del d['warnings']
            ttc = self.stats.vocab_type_counts(toks)
            # Only 1 per song
            (numer, denom), d['tempo'] = list(ttc['time_sig'].keys())[0], list(ttc['tempo'].keys())[0]
            d['time_sig'] = f'{numer}/{denom}'
            d['duration_count'], d['pitch_count'] = ttc['duration'], ttc['pitch']
            return d
        return pd.DataFrame([extract_info(e) for e in entries[:128]])  # TODO: debugging

    @staticmethod
    def _plot_wrapper(callback: Callable):
        plt.figure()
        callback()
        plt.show()

    def token_length_dist(self):
        def callback():
            sns.histplot(data=self.df, x='n_token', kde=True)
            plt.title('Histogram of #encoded tokens per song')
            plt.xlabel('#encoded tokens')
            plt.ylabel('count')
        MusicVisualize._plot_wrapper(callback)

    def time_sig_dist(self):
        def callback():
            sns.countplot(data=self.df, x='time_sig')
            plt.title('Histogram of time signature per song')
            plt.xlabel('Time Signature')
            plt.ylabel('count')
        MusicVisualize._plot_wrapper(callback)

    def tempo_dist(self):
        def callback():
            sns.histplot(data=self.df, x='tempo', kde=True)
            plt.title('Histogram of tempo per song')
            plt.xlabel('Tempo')
            plt.ylabel('count')
        MusicVisualize._plot_wrapper(callback)

    def note_pitch_dist(self):
        counts = Counter()
        for d in self.df.pitch_count:
            counts.update(d)
        df = pd.DataFrame([(k, v) for k, v in counts.items()], columns=['pitch', 'count'])
        ma, mi = df.pitch.max(), df.pitch.min()
        assert mi == -1
        ax = sns.histplot(
            data=df, x='pitch', weights='count', kde=True,
            bins=ma-mi+1, kde_kws=dict(bw_adjust=0.5)
        )
        pch_ints = [-1, *range(6, ma+6, 6)]
        ax.set_xticks(pch_ints, labels=[self.tokenizer.vocab.pitch_midi2name(p) for p in pch_ints])
        plt.title('Histogram of pitch across all songs')
        plt.xlabel('Pitch')
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


if __name__ == '__main__':
    from icecream import ic

    fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-25 20-59-06.json'
    fnm = os.path.join(config('path-export'), fnm)
    vs = MusicVisualize(fnm)

    def check_warn():
        df = vs.warn_info(as_counts=True)
        ic(df)
    # check_warn()

    # vs.token_length_dist()
    # vs.time_sig_dist()
    # vs.tempo_dist()
    vs.note_pitch_dist()
