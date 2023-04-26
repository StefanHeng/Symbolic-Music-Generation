"""
Moved a few single-use scripts here for productivity on more frequent scripts
"""

import os
import re
import glob
from os.path import join as os_join

from tqdm.auto import tqdm

from stefutil import *
from musicnlp.util import *
import musicnlp.util.music as music_util


logger = get_logger('Util Chore')


if __name__ == '__main__':
    # import music21 as m21
    # path_broken = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/broken/LMD-cleaned/broken'
    # # broken_fl = 'ABBA - I\'ve Been Waiting For You.mid'
    # # broken_fl = 'Aerosmith - Pink.3.mid'
    # broken_fl = 'Alice in Chains - Sludge Factory.mid'
    # # broken_fl = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/broken/LMD-cleaned/fixed/' \
    # #             'ABBA - I\'ve Been Waiting For You.band.mid'
    # mic(broken_fl)
    # scr = m21.converter.parse(os_join(path_broken, broken_fl))
    # mic(scr)

    def fix_delete_broken_files():
        import glob

        path_broken = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/LMD-cleaned_broken/*.mid'
        set_broken = set(clean_whitespace(stem(fnm)) for fnm in glob.iglob(path_broken))
        mic(set_broken)
        path_lmd_c = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/LMD-cleaned/*.mid'
        for fnm in glob.iglob(path_lmd_c):
            if stem(fnm) in set_broken:
                os.remove(fnm)
                set_broken.remove(stem(fnm))
                print('Deleted', fnm)
        mic(set_broken)
        assert len(set_broken) == 0, 'Not all broken files deleted'
    # fix_delete_broken_files()

    def fix_match_mxl_names_with_new_mid():
        path_lmd_v = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/LMD-cleaned_valid/*.mxl'
        mic(len(list(glob.iglob(path_lmd_v))))
        for fnm in glob.iglob(path_lmd_v):
            fnm_new = clean_whitespace(fnm)
            if fnm != fnm_new:
                os.rename(fnm, fnm_new)
                print(f'Renamed {pl.i(fnm)} => {pl.i(fnm_new)}')
    # fix_match_mxl_names_with_new_mid()

    def fix_ori_lmci_wrong_fnm():
        """
        Original LMCI midi filename change logic was wrong, duplicated filenames always starts with `version 0`

        Re-map using updated logic
        """
        dnm = 'LMCI'
        # dir_nm_to_modify = 'LMCI, broken copy'
        # dir_nm_to_modify = 'LMCI, LP'
        dir_nm_to_modify = 'LMCI, MS'
        correct_fnms = [stem(f) for f in music_util.get_converted_song_paths(dataset_name=dnm, fmt='mid', backend=None)]
        # mic(correct_fnms[:5])
        n_song = sconfig(f'datasets.{dnm}.meta.n_song')
        assert len(correct_fnms) == n_song

        pattern_converted = re.compile(r'^(?P<ordinal>\d{6})_(?P<fnm>.+)$')

        def get_ordinal(fnm: str) -> int:
            m = pattern_converted.match(fnm)
            assert m is not None
            return int(m.group('ordinal'))
        ordinal2correct_fnm = {get_ordinal(fnm): fnm for fnm in correct_fnms}

        # both `mid` and `mxl` files
        path_base = os_join(u.dset_path, 'converted', dir_nm_to_modify)
        it_mid = glob.iglob(os_join(path_base, '**/*.mid'), recursive=True)
        it_mxl = glob.iglob(os_join(path_base, '**/*.mxl'), recursive=True)
        paths_to_modify = sorted(chain_its([it_mid, it_mxl]))

        it = tqdm(paths_to_modify, desc='Fixing LMCI filenames', unit='song')
        n_rename = 0
        for path in it:
            ori_fnm = stem(path)
            ordinal = get_ordinal(ori_fnm)
            correct_fnm = ordinal2correct_fnm[ordinal]
            it.set_postfix(dict(ori_fnm=pl.i(ori_fnm), correct_fnm=pl.i(correct_fnm)))
            base_path = os.path.dirname(path)
            # mic(base_path)
            if ori_fnm != correct_fnm:
                ext = os.path.splitext(path)[1]
                assert ext in ('.mid', '.mxl')
                new_path = os_join(base_path, f'{correct_fnm}{ext}')
                os.rename(path, new_path)
                # mic(path, new_path)
                # raise NotImplementedError
                n_rename += 1
                ori_fnm, correct_fnm = f'{ori_fnm}{ext}', f'{correct_fnm}{ext}'
                logger.info(f'{pl.i(ori_fnm)} renamed to {pl.i(correct_fnm)}')
                # raise NotImplementedError
        logger.info(f'{pl.i(n_rename)} files renamed')
    # fix_ori_lmci_wrong_fnm()

    def find_missing_lmci_file():
        """
        From LMCI processing, 2 files were missing, not in converted and not in broken, find them
        """
        from tqdm.auto import trange

        dnm = 'LMCI'
        conv_paths = set(music_util.get_converted_song_paths(dataset_name=dnm, fmt='mxl', backend='all'))
        pattern_fl = re.compile(r'^(?P<ordinal>\d{6})_(?P<fnm>.+)$')
        ords_conv = {int(pattern_fl.match(stem(fnm)).group('ordinal')) for fnm in conv_paths}

        n_song = sconfig(f'datasets.{dnm}.meta.n_song')
        dir_nm = sconfig(f'datasets.{dnm}.converted.dir_nm')

        brk_base = os_join(get_base_path(), u.dset_dir, f'{dir_nm}, broken')
        brk_paths = set(glob.iglob(os_join(brk_base, '**/*.mid'), recursive=True))
        ords_brk = {int(pattern_fl.match(stem(fnm)).group('ordinal')) for fnm in brk_paths}

        mic(len(conv_paths), len(brk_paths), n_song)

        if len(conv_paths) + len(brk_paths) != n_song:
            for i in trange(n_song, desc='Finding missing file'):
                if i not in ords_conv and i not in ords_brk:
                    logger.info(f'Missing file with ordinal {pl.i(i)}')
    find_missing_lmci_file()
