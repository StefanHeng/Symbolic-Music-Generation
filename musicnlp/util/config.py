import os
import re
import glob
from os.path import join as os_join
from typing import Dict, List, Union

import pandas as pd

from stefutil import *
from musicnlp.util.project_paths import BASE_PATH, DSET_DIR


config_dict: dict = {
    'datasets': {
        # for LMD datasets, see https://colinraffel.com/projects/lmd/
        'LMD': dict(
            nm='The Lakh MIDI Dataset, Full',
            original=dict(
                dir_nm='original/lmd_full',
                song_fmt_mid='**/*.mid',
            ),
            converted=dict(
                dir_nm='converted/LMD',
                song_fmt_mid='**/*.mid',
                song_fmt_mxl='**/*.mxl'
            )
        ),
        'LMD-matched': dict(
            nm='The Lakh MIDI Dataset, Matched',
            original=dict(
                dir_nm='original/Lakh-MIDI-Dataset/LMD-Matched'
            )
        ),
        'LMD-aligned': dict(
            nm='The Lakh MIDI Dataset, Aligned',
            original=dict(
                dir_nm='original/Lakh-MIDI-Dataset/LMD-Aligned'
            )
        ),
        'LMD-cleaned': dict(
            nm='The Lakh MIDI Dataset, Cleaned',
            original=dict(
                dir_nm='original/Lakh-MIDI-Dataset/LMD-Cleaned',
                song_fmt_mid='**/*.mid',
            ),
            converted=dict(song_fmt_mid='*.mid', song_fmt_mxl='*.mxl')
        ),
        'LMD-cleaned-subset': dict(
            nm='Single-Song-Version Subset of LMD-Cleaned',
            converted=dict(
                song_fmt_mid='*.mid',
                song_fmt_mxl='*.mxl',
                dir_nm='converted/LMD-cleaned-subset'
            )
        ),
        'MAESTRO': dict(
            nm='The MAESTRO Dataset v3.0.0',
            original=dict(
                dir_nm='original/maestro-v3.0.0',
                song_fmt_mid='**/*.midi',
            ),
            converted=dict(
                song_fmt_mid='*.mid',
                song_fmt_mxl='*.mxl',
                dir_nm='converted/MAESTRO'
            )
        ),
        'POP909': dict(
            nm='POP909 Dataset for Music Arrangement Generation',
            dir_nm='original/POP909-Dataset',
            original=dict(song_fmt_mid='*.mid'),
            converted=dict(
                song_fmt_mid='*.mid',
                song_fmt_mxl='*.mxl',
                dir_nm='converted/POP909'
            )
        ),
        'LMCI': dict(
            # See https://www.reddit.com/r/datasets/comments/3akhxy/the_largest_midi_collection_on_the_internet/
            nm='Largest MIDI Collection on the Internet',
            original=dict(
                dir_nm='original/MIDI_Archive_15-06-19',
                song_fmt_mid=['**/*.mid', '**/*.midi']
            ),
            converted=dict(
                dir_nm='converted/LMCI',
                song_fmt_mid='**/*.mid',
                song_fmt_mxl='**/*.mxl'
            )
        ),
        'NES-MDB': dict(
            # See https://github.com/chrisdonahue/nesmdb
            nm='NES Music Database',
            original=dict(
                dir_nm='original/nesmdb_midi',
                song_fmt_mid='**/*.mid',
            ),
            converted=dict(
                dir_nm='converted/NES-MDB',
                song_fmt_mid='*.mid',
                song_fmt_mxl='*.mxl'
            )
        ),
        'midi-eg': dict(
            nm='Some hand-selected MIDI samples',
            dir_nm='MIDI-eg',
            song_fmt_mid='*.mid'
        ),
        'mxl-eg': dict(
            nm='Some hand-selected MXL samples',
            dir_nm='MXL-eg',
            dir_nm_extracted='MXL-eg_out',
            song_fmt_mxl='*.mxl'
        ),
        'my': dict(
            nm='Music with NLP, Project output',
            dir_nm='processed'
        )
    },
    'random-seed': 77,
    'check-arg': [
        dict(
            display_name='Song Representation/Extraction Mode', attr_name='extract_mode',
            accepted_values=['melody', 'full']
        ),
        dict(
            display_name='Extraction Export Type', attr_name='exp',
            accepted_values=['mxl', 'str', 'id', 'str_join', 'visualize']
        ),
        dict(display_name='Music File Format', attr_name='fmt', accepted_values=['mid', 'mxl']),
        dict(display_name='Music File Conversion Backend', attr_name='backend', accepted_values=['MS', 'LP']),

        dict(display_name='Music Key Type', attr_name='key_type', accepted_values=['list', 'enum', 'dict']),

        dict(
            display_name='Music Channel Mixup Augmentation', attr_name='channel_mixup', accepted_values=['full', 'swap']
        ),

        dict(display_name='Train Logging Strategy', attr_name='log_strategy', accepted_values=['epoch', 'steps', 'no']),
        dict(display_name='Train Logging Mode', attr_name='log_mode', accepted_values=['train', 'eval']),
        dict(
            display_name='Generation Mode', attr_name='generation_mode',
            accepted_values=['conditional', 'unconditional']
        ),
        dict(
            display_name='Generation Strategy', attr_name='generation_strategy',
            accepted_values=['greedy', 'sample', 'contrastive', 'beam']
        ),
        dict(
            display_name='Distribution Plot Type', attr_name='dist_plot_type',
            accepted_values=['bar', 'hist']
        )
    ]
}


for k in it_keys(config_dict['datasets']):    # Accommodate other OS
    k = f'{DSET_DIR}.{k}'
    val = get(config_dict, k)
    if k[k.rfind('.')+1:] == 'dir_nm':
        set_(config_dict, k, os_join(*val.split('/')))  # string are in macOS separator


def get_stats(songs: List[Dict]):
    return dict(
        n_file=len(songs),
        n_artist=len(set(s['artist'] for s in songs)),
        n_song=len(set((s['artist'], s['title']) for s in songs))  # in case same title by different artists
    )


def get_dataset_meta(dataset_name: str):
    ca.check_mismatch('Dataset Name', dataset_name, ['POP909', 'LMD-cleaned', 'LMD', 'LMCI', 'MAESTRO', 'NES-MDB'])
    if dataset_name in ['LMD', 'LMCI', 'MAESTRO', 'NES-MDB']:
        d_dset = get(config_dict, f'datasets.{dataset_name}.original')
        path_ori = os_join(BASE_PATH, DSET_DIR, d_dset['dir_nm'])

        if dataset_name == 'LMCI':
            exts = ('.mid', '.midi')  # some are in upper case
            n = len(set(p for p in glob.iglob(os_join(path_ori, '**/*'), recursive=True) if p.lower().endswith(exts)))
        else:  # `LMD`, `MAESTRO`, `NES-MDB`
            n = len(set(glob.iglob(os_join(path_ori, d_dset['song_fmt_mid']), recursive=True)))
        return dict(n_song=n)
    else:
        if dataset_name == 'POP909':
            path = os_join(BASE_PATH, DSET_DIR, 'original/POP909-Dataset', dataset_name)  # the original path
            df = pd.read_excel(os_join(path, 'index.xlsx'))

            def map_single(d: Dict):
                return dict(artist=d['artist'], title=d['name'])
            songs = [map_single(d) for d in df.T.to_dict().values()]
        else:
            assert dataset_name == 'LMD-cleaned'
            d_dset = get(config_dict, f'datasets.{dataset_name}.original')
            path_ori = os_join(BASE_PATH, DSET_DIR, d_dset['dir_nm'])

            pattern_title = re.compile(r'^(?P<title>.*)\.(?P<version>[1-9]\d*).mid$')  # <title>.<version>.mid

            def clean_title(title: str) -> Union[bool, str]:
                """
                Strip potential version information

                :return: A string of the cleaned title, or False if it doesn't need to be cleaned
                """
                m = pattern_title.match(title)
                return f'{m.group("title")}.mid' if m else False

            def path2song(path_: str):
                paths_last = path_.split(os.sep)[-2:]
                artist, title = paths_last[0], paths_last[1]
                title_cleaned = clean_title(title)
                if title_cleaned:
                    return dict(artist=artist, title=title_cleaned, original_title=title)
                else:
                    return dict(artist=artist, title=title)
            songs = [path2song(p_) for p_ in glob.glob(os_join(path_ori, d_dset['song_fmt_mid']))]
        songs = sorted(songs, key=lambda s: (s['artist'], s['title']))  # sort by artist, then title
        # return dict(songs=songs) | get_stats(songs)
        return get_stats(songs)


for dnm in ['POP909', 'LMD-cleaned', 'LMD', 'LMCI', 'MAESTRO', 'NES-MDB']:
    set_(config_dict, f'datasets.{dnm}.meta', get_dataset_meta(dnm))


config_dict['check-arg'].append(dict(
    display_name='Dataset Name', attr_name='dataset_name',
    accepted_values=list(config_dict['datasets'].keys()) + ['LMD-cleaned-subset']
))


if __name__ == '__main__':
    import json
    from project_paths import PROJ_DIR, PKG_NM

    mic.output_width = 256

    fl_nm = 'config.json'
    mic(config_dict)
    # print(config)
    open(fl_nm, 'a').close()  # Create file in OS
    with open(os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', fl_nm), 'w') as f:
        json.dump(config_dict, f, indent=4)
