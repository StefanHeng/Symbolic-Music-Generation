from tqdm import tqdm
import datasets

from musicnlp.util import *
import musicnlp.util.music as music_util
from music_extractor import MusicExtractor


class MusicExport:
    """
    Batch export extracted/tokenized music from `MusicTokenizer` in a more accessible format
    """
    def __init__(self, mode='melody', verbose: Union[bool, str] = False):
        """
        :param mode: One of [`melody`, `full`], see `MusicTokenizer`
            TODO: support chords in MusicTokenizer
        :param verbose: Arguments to `MusicExtractor`
        """
        self.verbose = verbose
        self.mode = mode

    def __call__(
            self,
            fnms: Union[List[str], str],
            fnm_out=f'{PKG_NM} music extraction', path_out=get_processed_path(),
            prec: int = 5,
            mode='melody',
            exp='str_join',
            parallel: Union[bool, int] = False,
            disable_tqdm: bool = False
    ):
        """
        Writes encoded files to JSON file

        :param fnms: List of MXL file paths to extract, without `.json` extension;
            or dataset name, see `config.datasets`
        :param fnm_out: Export file name
        :param mode: Music extraction mode, see `MusicTokenizer`
        :param exp: Music extraction output mode, see `MusicTokenizer`
        :param parallel: Whether to parallelize extraction
            If true, a batch size may be specified
        """
        exp_opns = ['str', 'id', 'str_join']
        if exp not in exp_opns:
            raise ValueError(f'Unexpected export mode - got {logi(exp)}, expect one of {logi(exp_opns)}')

        dnm_ = None
        if isinstance(fnms, str):  # Dataset name provided
            dnm_ = fnms
            fnms = music_util.get_cleaned_song_paths(fnms, fmt='mxl')[:20]  # TODO: debugging
            # so that warnings are for each song
        log(f'Extracting {logi(len(fnms))} songs... ')
        me_ = MusicExtractor(precision=prec, mode=self.mode, warn_logger=True, verbose=self.verbose, save_memory=True)

        def call_single(fl_nm) -> Dict:
            return me_(fl_nm, exp=exp, return_meta=True)

        if parallel:
            def batched_map(fnms_, s, e):
                return [call_single(fnms_[i]) for i in range(s, e)]
            lst_out = batched_conc_map(batched_map, fnms, batch_size=(isinstance(parallel, int) and parallel) or 32)
        else:
            lst_out = []
            gen = enumerate(fnms)
            if not disable_tqdm:
                gen = tqdm(gen, total=len(fnms), desc='Extracting music', unit='song')
            for i_fl, fnm in gen:
                lst_out.append(call_single(fnm))
        if dnm_ is not None:
            fnm_out += f', dnm={dnm_}'
        fnm_out += f', n={len(fnms)}, mode={mode}, {now(for_path=True)}'
        with open(os.path.join(path_out, f'{fnm_out}.json'), 'w') as f:
            # TODO: Knowing the extracted dict, expand only the first few levels??
            json.dump(dict(precision=prec, encoding_type=exp, music=lst_out), f, indent=4)

    @staticmethod
    def json2dataset(fnm: str, path_out=get_processed_path()) -> datasets.Dataset:
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

    # me = MusicExport()
    # me = MusicExport(verbose=True)
    me = MusicExport(verbose='single')

    def check_parallel():
        me('LMD-cleaned-subset', parallel=3)
    check_parallel()

    def export2json():
        # dnm = 'POP909'
        dnm = 'LMD-cleaned-subset'
        # me(dnm)
        me(dnm, parallel=64)
    # export2json()

    def json2dset():
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-22 19-00-40'
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-25 20-59-06'
        fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-03-01 02-29-29'
        dset = me.json2dataset(fnm)
        ic(dset, dset[:5])
    # json2dset()
