from tqdm import tqdm
import datasets

from musicnlp.util import *
from music_extractor import MusicExtractor


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
            prec: int = 5,
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
            fnms = get_cleaned_song_paths(fnms, fmt='song_fmt_exp')
        lst_out = []
        me_ = MusicExtractor(precision=prec, mode=self.mode, logger=True, verbose=self.verbose)
        for i_fl, fnm in tqdm(enumerate(fnms), total=len(fnms)):
            if self.verbose:
                log(f'Extracting file {logi(stem(fnm))}... ')
            txt, secs = me_(fnm, exp=exp, return_duration=True)
            lst_out.append(dict(
                title=me_.title,  # No parallelism, title of current processed music file
                text=txt, warnings=me_.logger.tracked(exp='serialize'),
                duration=secs
            ))
        if dnm_ is not None:
            fnm_out += f', dnm={dnm_}'
        fnm_out += f', n={len(fnms)}, mode={mode}, {now(for_path=True)}'
        with open(os.path.join(path_out, f'{fnm_out}.json'), 'w') as f:
            # TODO: Knowing the extracted dict, expand only the first few levels??
            json.dump(dict(precision=prec, encoding_type=exp, music=lst_out), f, indent=4)

    @staticmethod
    def json2dataset(fnm: str, path_out=config('path-export')) -> datasets.Dataset:
        """
        Save extracted `.json` dataset by `__call__`, as HuggingFace Dataset to disk
        """
        with open(os.path.join(path_out, f'{fnm}.json')) as f:
            dset_: Dict = json.load(f)
        tr = dset_['music']  # TODO: All data as training?

        def prep_entry(d: Dict) -> Dict:
            del d['warnings']
            return d
        dset = datasets.Dataset.from_pandas(
            pd.DataFrame([prep_entry(d) for d in tr]),
            info=datasets.DatasetInfo(description=json.dumps(dict(precision=dset_['precision'])))
        )
        dset.save_to_disk(os.path.join(path_out, 'hf_datasets', fnm))
        return dset


if __name__ == '__main__':
    from icecream import ic

    me = MusicExport()

    def export2json():
        dnm = 'POP909'
        me(dnm)
    # export2json()

    def json2dset():
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody,  2022-02-22 19-00-40'
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-25 20-59-06'
        fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-03-01 02-29-29'
        dset = me.json2dataset(fnm)
        ic(dset, dset[:5])
    json2dset()
