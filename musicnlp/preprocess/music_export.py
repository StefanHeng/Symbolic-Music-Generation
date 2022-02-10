from tqdm import tqdm

from musicnlp.util import *
from music_extractor import MusicTokenizer


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
            fnms: List[str],
            fnm_out=f'{PKG_NM} music extraction', path_out=config('path-export'),
            exp='str_join'
    ):
        """
        Writes encoded files to JSON file

        :param fnms: List of MXL file paths to extract, without `.json` extension
        :param fnm_out: Export file name
        :param exp: Music extraction output mode, see `MusicTokenizer`
        """
        lst_out = []
        mt = MusicTokenizer(mode=self.mode, logger=True, verbose=self.verbose)
        for i_fl, fnm in tqdm(enumerate(fnms)):
            if self.verbose:
                log(f'Extracting file {logi(stem(fnm))}... ')
            out = mt(fnm, exp=exp)
            lst_out.append(dict(title=mt.title, out=out, warnings=mt.logger.tracked(summary=False)))  # Current title
        with open(os.path.join(path_out, f'{fnm_out}-{now(sep="-")}.json'), 'w') as f:
            json.dump(dict(out_type=exp, music=lst_out), f, indent=4)


if __name__ == '__main__':
    from icecream import ic

    dnm = 'POP909'
    fnms_ = fl_nms(dnm, k='song_fmt_exp')

    me = MusicExport()
    me(fnms=fnms_[:5])
