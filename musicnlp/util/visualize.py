

from musicnlp.util import *
from musicnlp.model import LMTTokenizer


class Visualize:
    """
    Visualize dataset info given json as extracted input representation

    See `preprocess.music_export.py`
    """
    def __init__(self, fl_nm):
        with open(os.path.join(fl_nm)) as f:
            self.dset: Dict = json.load(f)

        self.tokenizer = LMTTokenizer(prec=self.dset['precision'])

    def _get_song_info(self):
        songs: List[Dict] = self.dset['music']

        def extract_info(d: Dict):
            toks = d.pop('text').tokenize()
            d['n_token'] = len(toks)
            d['time_sig'] = self.tokenizer.vocab.compact()

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
    vs = Visualize(fnm)

    def check_warn():
        df = vs.warn_info(as_counts=True)
        ic(df)
    check_warn()
