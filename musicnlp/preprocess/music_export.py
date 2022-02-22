from tqdm import tqdm
import datasets

from musicnlp.util import *
from music_extractor import WarnLog, MusicTokenizer


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

    def json2dataset(self, fnm: str, path_out=config('path-export')) -> datasets.Dataset:
        """
        Save extracted `.json` dataset by `__call__`, as HuggingFace Dataset to disk
        """
        with open(fnm) as f:
            dset: List[Dict] = json.load(f)

    def json2warn_df(self, fnm: str, path_out=config('path-export')) -> pd.DataFrame:
        """
        Aggregate warnings as a pandas Dataframe
        """
        d_warn = {t: None for t in WarnLog.TYPES}

        def d2d_exp(d: Dict) -> Dict:
            # Flatten for compatible with 2d dataset
            # TODO: __call__ API change
            d['text'] = d.pop('out', None)
            d |= d_warn


if __name__ == '__main__':
    from icecream import ic

    me = MusicExport()

    def export2json():
        dnm = 'POP909'
        me(dnm)
    # export2json()

    def json2dset():
        fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody,  2022-02-20 12-17-05.json'
        me.json2dataset(fnm)
    json2dset()

