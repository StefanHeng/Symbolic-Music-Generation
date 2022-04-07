import glob

from util import *


d_allie = dict(
    dir_nm='Allie-Chord-Embedding',
    nm='Allie-Chord-Embedding',
    nm_data='full_song_objects.pickle'
)
fnm = os.path.join(PATH_BASE, DIR_DSET, d_allie['dir_nm'], d_allie['nm_data'])
d_allie['n_entry'] = len(read_pickle(fnm)[0])


config: dict = {
    'datasets': {
        'Allie-Chords': d_allie,
        # for LMD datasets, see https://colinraffel.com/projects/lmd/
        'LMD-matched': dict(
            nm='The Lakh MIDI Dataset, Matched',
            dir_nm='Lakh-MIDI-Dataset/LMD-Matched'
        ),
        'LMD-aligned': dict(
            nm='The Lakh MIDI Dataset, Aligned',
            dir_nm='Lakh-MIDI-Dataset/LMD-Aligned'
        ),
        'LMD-cleaned': dict(
            nm='The Lakh MIDI Dataset, Cleaned',
            dir_nm='Lakh-MIDI-Dataset/LMD-Cleaned',
            song_fmt_mid='**/*.mid',
            song_fmt_mxl='**/*.mxl'
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
            song_fmt_mid='*.mxl'
        ),
        'POP909': dict(
            nm='POP909 Dataset for Music Arrangement Generation',
            dir_nm='POP909',
            song_fmt_mid='*.mid',
            song_fmt_mxl='*.mxl'
        ),
        'my': dict(
            nm='Music with NLP, Project output',
            dir_nm='MNLP-Combined'
        )
    },
    'random-seed': 77,
}


for k in keys(config[DIR_DSET]):    # Accommodate other OS
    k = f'{DIR_DSET}.{k}'
    val = get(config, k)
    if k[k.rfind('.')+1:] == 'dir_nm':
        set_(config, k, os.path.join(*val.split('/')))


def get_stats(songs: List[Dict]):
    return dict(
        n_file=len(songs),
        n_artist=len(set(s['artist'] for s in songs)),
        n_song=len(set((s['artist'], s['title']) for s in songs))  # in case same title by different artists
    )


def get_dataset_meta(dataset_name: str):
    dnms = ['POP909', 'LMD-cleaned']
    assert dataset_name in dnms, f'Unsupported dataset name: expect one of {logi(dnms)}, got {logi(dataset_name)}'
    if dataset_name == 'POP909':
        path = os.path.join(PATH_BASE, DIR_DSET, 'POP909-Dataset', dataset_name)  # the original path
        df = pd.read_excel(os.path.join(path, 'index.xlsx'))

        def map_single(d: Dict):
            return dict(artist=d['artist'], title=d['name'])
        songs = [map_single(d) for d in df.T.to_dict().values()]
    else:
        d_dset = config['datasets'][dataset_name]
        path_ori = os.path.join(PATH_BASE, DIR_DSET, d_dset['dir_nm'])

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
        songs = [path2song(p_) for p_ in glob.glob(os.path.join(path_ori, d_dset['song_fmt_mid']))]
    songs = sorted(songs, key=lambda s: (s['artist'], s['title']))  # sort by artist, then title
    return dict(songs=songs) | get_stats(songs)


for dnm in ['POP909', 'LMD-cleaned']:
    config['datasets'][dnm]['meta'] = get_dataset_meta(dnm)


if __name__ == '__main__':
    import json
    from data_path import *

    from icecream import ic

    fl_nm = 'config.json'
    # ic(config)
    # print(config)
    open(fl_nm, 'a').close()  # Create file in OS
    with open(os.path.join(PATH_BASE, DIR_PROJ, PKG_NM, 'util', fl_nm), 'w') as f:
        json.dump(config, f, indent=4)
