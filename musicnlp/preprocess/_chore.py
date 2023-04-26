"""
Moved a few single-use scripts here for productivity on more frequent scripts
"""

import os
import json
import glob
from os.path import join as os_join

from tqdm.auto import tqdm

from stefutil import *
from musicnlp.util import *
import musicnlp.util.music as music_util
from musicnlp.preprocess import MusicExport


if __name__ == '__main__':
    mic.output_width = 512

    """
    Music Export Chores
    """
    me = MusicExport()

    def check_export_json_error():
        dnm = 'MAESTRO'
        dir_nm = sconfig(f'datasets.{dnm}.converted.dir_nm')
        dir_nm = f'{dir_nm}, MS'
        # fnm = 'Franz Liszt - Après Une Lecture De Dante: Fantasia Quasi Sonata, S.161, No. 7.mxl'
        # fnm = 'Johann Sebastian Bach - Prelude And Fugue In E Major, Wtc I, Bwv 854.mxl'
        fnm = 'Franz Liszt - Études D\'exécution Transcendante, No. 5, "feux Follets" S.139:5.mxl'
        path = os_join(u.dset_path, dir_nm, fnm)
        me([path], extractor_args=dict(greedy_tuplet_pitch_threshold=1))
    # check_export_json_error()


    def fix_insert_key():
        """
        As extracting from all the 10k songs is time-consuming,
        temporarily insert keys into the already processed json files
        """
        from musicnlp.preprocess.key_finder import KeyFinder

        # dnm = 'POP909'
        dnm = 'LMD-cleaned-subset'
        # dir_nm = 'POP909 save single 04-10_02.15'
        dir_nm = 'LMD-cleaned_subset save single 04-09_21-51'
        dir_nm_out = f'{dir_nm}, add key'
        path = os_join(music_util.get_processed_path(), 'intermediate', dir_nm)
        path_out = os_join(music_util.get_processed_path(), 'intermediate', dir_nm_out)
        os.makedirs(path_out, exist_ok=True)
        fnms = sorted(glob.iglob(os_join(path, '*.json')))

        nm = 'Insert Key Back'
        logger = get_logger(nm)
        pbar = tqdm(total=len(fnms), desc=nm, unit='song')

        dir_nm_dset = 'LMD-cleaned_valid' if dnm == 'LMD-cleaned-subset' else dnm  # for pop909

        def song_title2path(title: str) -> str:
            # Needed cos the original json files may not be processed on my local computer
            return os_join(BASE_PATH, DSET_DIR, dir_nm_dset, f'{title}.mxl')

        def call_single(fl_nm: str):
            try:
                fnm_out = os_join(path_out, f'{stem(fl_nm)}.json')
                if not os.path.exists(fnm_out):
                    with open(fl_nm, 'r') as f:
                        song = json.load(f)
                    # mic(song.keys())
                    assert 'keys' not in song['music']  # sanity check
                    # mic(fl_nm, song.keys(), song['music'].keys())
                    path_mxl = song_title2path(song['music']['title'])
                    song['music']['keys'] = keys = KeyFinder(path_mxl)(return_type='dict')
                    assert len(keys) > 0
                    with open(fnm_out, 'w') as f:
                        json.dump(song, f, indent=4)
                pbar.update(1)
            except Exception as e:
                logger.error(f'Failed to find key for {pl.i(fl_nm)}, {pl.i(e)}')  # Abruptly stop the process
                raise ValueError(f'Failed to find key for {pl.i(fl_nm)}')
        # for fnm in fnms[:20]:
        #     call_single(fnm)

        def batched_map(fnms_, s, e):
            return [call_single(fnms_[i]) for i in range(s, e)]
        n_worker = os.cpu_count() * 2  # majority of the time is m21 parsing file
        batched_conc_map(batched_map, fnms, batch_size=32, n_worker=n_worker)
    # fix_insert_key()
    # profile_runtime(fix_insert_key)

    def fix_key_api_change():
        """
        Original KeyFinder results are stored in a key `key`, the new API uses `keys`, so update the written files
        """
        dir_nm = 'LMD-cleaned_subset save single 04-09_21-51, add key'
        # dir_nm = 'POP909 save single 04-10_02.15, add key'
        path = os_join(music_util.get_processed_path(), 'intermediate', dir_nm)
        fnms = sorted(glob.iglob(os_join(path, '*.json')))
        # mic(len(fnms))
        # exit(1)

        count_old_key = 0
        for fnm in tqdm(fnms):
            with open(fnm, 'r') as f:
                d = json.load(f)
            if 'key' in d['music']:
                d['music']['keys'] = d['music'].pop('key')
                count_old_key += 1
                with open(fnm, 'w') as f:  # Override the original file
                    json.dump(d, f, indent=4)
        mic(count_old_key)
    # fix_key_api_change()

    def fix_wrong_moved_fnm():
        import shutil
        dir_nm = '2022-05-20_09-39-16_LMD, MS'
        path_process_base = os_join(u.dset_path, 'processed', 'intermediate', dir_nm)
        path_to_process = os_join(path_process_base, '100000-110000')
        paths = sorted(glob.iglob(os_join(path_to_process, '*.json'), recursive=True))
        for path in tqdm(paths):
            fnm = stem(path)[-11:-5]
            path_new = os_join(path_to_process, f'Music Export - {fnm}.json')
            # mic(fnm, path, path_new)
            # exit(1)
            shutil.move(path, path_new)
    # fix_wrong_moved_fnm()

    def fix_move_folder_lmci_name_wrong():
        """
        In `chore_move_proper_folder` on LMCI dataset, the postfix original filename is missing,
        rename those files to correct name
        """
        import re

        # dnm = 'LMCI'
        dir_nm_conv = 'LMCI, MS'
        dir_nm_proc = '23-04-18_LMCI_{md=f}'
        grp_nm = '090000-100000'

        # get ordinal to correct filename mapping

        glob_pat = os_join(u.dset_path, 'converted', dir_nm_conv, grp_nm, '*.mxl')
        paths_correct = sorted(glob.iglob(glob_pat, recursive=True))
        pattern_correct = re.compile(r'^(?P<ordinal>\d*)_(?P<name>.*)$')

        def correct_fnm2ordinal(fnm_):
            return int(pattern_correct.match(fnm_).group('ordinal'))
        ord2fnm = dict()
        for path in tqdm(paths_correct, desc='Getting Correct Name'):
            fnm = stem(path)
            ord2fnm[correct_fnm2ordinal(fnm)] = fnm

        path_base = os_join(music_util.get_processed_path(), 'intermediate', dir_nm_proc, grp_nm)
        paths = sorted(glob.iglob(os_join(path_base, '*.json'), recursive=True))

        pattern_broken = re.compile(r'^Music Export - (?P<ordinal>\d*)$')
        for path in tqdm(paths, 'Renaming'):
            # mic(path)
            m = pattern_broken.match(stem(path))
            if m is not None:  # otherwise, already run
                o = int(m.group('ordinal'))

                fnm_correct = ord2fnm[o]
                path_new = os_join(path_base, f'Music Export - {fnm_correct}.json')
                # rename file
                os.rename(path, path_new)
    # fix_move_folder_lmci_name_wrong()
