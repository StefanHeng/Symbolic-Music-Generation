from icecream import ic
from util import *


d_allie = dict(
    dir_nm='Allie-Chord-Embedding',
    nm='Allie-Chord-Embedding',
    nm_data='full_song_objects.pickle'
)
fnm = f'../datasets/{d_allie["dir_nm"]}/{d_allie["nm_data"]}'
d_allie['n_entry'] = len(read_pickle(fnm)[0])


config = dict(
    datasets=dict(
        Allie_Chords=d_allie,
        LMD_matched=dict(
            dir_nm='Lakh-MIDI-Dataset/LMD-Matched',
            nm='LMD-Matched'
        ),
        LMD_Aligned=dict(
            dir_nm='Lakh-MIDI-Dataset/LMD-Aligned',
            nm='LMD-Aligned'
        )
    )
)

if __name__ == '__main__':
    import json
    from data_path import *

    fl_nm = 'config.json'
    ic(config)
    open(fl_nm, 'a').close()  # Create file in OS
    with open(f'{PATH_BASE}/{fl_nm}', 'w') as f:
        json.dump(config, f, indent=4)

