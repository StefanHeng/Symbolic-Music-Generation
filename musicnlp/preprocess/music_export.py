import pandas as pd
from tqdm import tqdm
import datasets

from musicnlp.util import *
from music_extractor import MusicTokenizer


pd.set_option('expand_frame_repr', False)
pd.set_option('display.precision', 2)
# pd.set_option('display.max_colwidth', 50)
# pd.set_option('display.max_columns', 50)
# pd.set_option('display.max_info_columns', 50)


class MusicExport:
    """
    Batch export extracted/tokenized music from `MusicTokenizer` in a more accessible format
    """
    def __init__(self, mode='melody', verbose=False):
        """
        :param mode: One of [`melody`, `full`], see `MusicTokenizer`
            TODO: support chords in MusicTokenizer
        :param verbose: Arguments to `MusicTokenizer`
        """
        self.verbose = verbose
        self.mode = mode

    def __call__(
            self,
            fnms: Union[List[str], str],
            fnm_out=f'{PKG_NM} music extraction', path_out=config('path-export'),
            mode='melody',
            exp='str_join'
    ):
        """
        Writes encoded files to JSON file

        :param fnms: List of MXL file paths to extract, without `.json` extension;
            or dataset name, see `config.datasets`
        :param fnm_out: Export file name
        :param mode: Music extraction mode, see `MusicTokenizer`
        :param exp: Music extraction output mode, see `MusicTokenizer`
        """
        dnm_ = None
        if isinstance(fnms, str):  # Dataset name provided
            dnm_ = fnms
            fnms = fl_nms(fnms, k='song_fmt_exp')
        lst_out = []
        mt = MusicTokenizer(mode=self.mode, logger=True, verbose=self.verbose)
        for i_fl, fnm in tqdm(enumerate(fnms), total=len(fnms)):
            if self.verbose:
                log(f'Extracting file {logi(stem(fnm))}... ')
            txt = mt(fnm, exp=exp)
            lst_out.append(dict(
                title=mt.title,  # No parallelism, title of current processed music file
                text=txt, warnings=mt.logger.tracked(exp='serialize')
            ))
        if dnm_ is not None:
            fnm_out += f', dnm={dnm_}'
        fnm_out += f', n={len(fnms)}, mode={mode},  {now(sep="-")}'
        with open(os.path.join(path_out, f'{fnm_out}.json'), 'w') as f:
            # TODO: Knowing the extracted dict, expand only the first few levels??
            json.dump(dict(encoding_type=exp, music=lst_out), f, indent=4)

    @staticmethod
    def json2dataset(fnm: str, path_out=config('path-export')) -> datasets.Dataset:
        """
        Save extracted `.json` dataset by `__call__`, as HuggingFace Dataset to disk
        """
        with open(os.path.join(path_out, fnm)) as f:
            dset_: Dict = json.load(f)
        train = dset_['music']  # All data as training?

        def prep_entry(d: Dict) -> Dict:
            d['text'], _ = d.pop('out', None), d.pop('warnings', None)  # Strip warnings
            return d
        return datasets.Dataset.from_pandas(pd.DataFrame([prep_entry(d) for d in train]))

    @staticmethod
    def json2warn_df(fnm: str, path_out=config('path-export')) -> pd.DataFrame:
        """
        Aggregate warnings as a pandas Dataframe
        """
        with open(os.path.join(path_out, fnm)) as f:
            dset_: Dict = json.load(f)
        entries = dset_['music']

        def entry2df(d: Dict) -> pd.DataFrame:
            # Flatten for compatible with 2d dataset
            # TODO: __call__ API change
            warns, title = d['warnings'], d['title']

            def prep_warn(d_warn: Dict) -> Dict:
                d_out = dict()
                d_out['src'] = title
                d_out['type'] = d_warn.pop('warn_name', None)
                d_out['args'] = json.dumps(d_warn)
                return d_out
            return pd.DataFrame([prep_warn(d) for d in warns])
        return pd.concat([entry2df(e) for e in entries])


def warn_df2stats(df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """
    Get statistics about warnings logged

    Average warning, per song
    """
    # ic(df.nunique())
    # df = df.groupby('type')['type'].nunique()
    # ic(df)
    n_song = df.src.nunique()
    # ic(n_song)
    counts = df.type.value_counts()
    # ic(counts, type(counts), counts.to_dict())
    df = counts.to_frame(name='total_count').reset_index()
    df.rename(columns={'index': 'type'}, inplace=True)
    # ic(df, df.columns)
    df['average_count'] = df.apply(lambda x: x.total_count / n_song, axis=1)
    # ic(df)
    return n_song, df


if __name__ == '__main__':
    from icecream import ic

    me = MusicExport()

    def export2json():
        dnm = 'POP909'
        me(dnm)
    # export2json()

    def json2dset():
        fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody,  2022-02-20 12-17-05.json'
        dset = me.json2dataset(fnm)
        ic(dset, dset[:5])
    # json2dset()

    def json2warn():
        fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody,  2022-02-20 12-17-05.json'
        df = me.json2warn_df(fnm)
        ic(df)

        warn_df2stats(df)
    json2warn()

