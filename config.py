import os

from icecream import ic

from util import *


d_allie = dict(
    dir_nm='Allie-Chord-Embedding',
    nm='Allie-Chord-Embedding',
    nm_data='full_song_objects.pickle'
)
fnm = f'../datasets/{d_allie["dir_nm"]}/{d_allie["nm_data"]}'
d_allie['n_entry'] = len(read_pickle(fnm)[0])


def get_tokenizer_attr():
    # should be a one-to-one map
    n_special = 2 ** 7
    half = n_special / 2
    vocab_enc = {
        '[SEP]': 0,  # Bar separation
        '[TRIP]': 1,  # Last quarter encoding for triplets
        '[REST]': int(half)
    }
    # pitch midi follows after `n_special`
    vocab_enc.update({pitch: pitch+n_special for pitch in range(2**7)})  # Per MIDI spec
    # ic(vocab_enc)
    vocab_dec = {v: k for k, v in vocab_enc.items()}
    # ic(vocab_dec)
    return dict(
        n_special_token=n_special,
        vocab_enc=vocab_enc,
        vocab_dec=vocab_dec
    )


config = {
    DIR_DSET: dict(
        Allie_Chords=d_allie,
        LMD_matched=dict(
            nm='The Lakh MIDI Dataset, Matched',
            dir_nm='Lakh-MIDI-Dataset/LMD-Matched',
        ),
        LMD_Aligned=dict(
            nm='The Lakh MIDI Dataset, Aligned',
            dir_nm='Lakh-MIDI-Dataset/LMD-Aligned',
        ),
        MIDI_EG=dict(
            nm='Some hand-selected MIDI samples',
            dir_nm='MIDI-eg',
            fmt_song='*.mid'
        ),
        MXL_EG=dict(
            nm='Some hand-selected MXL samples',
            dir_nm='MXL-eg',
            fmt_song='*.mxl'
        )
    ),
    'Melody-Extraction': dict(
        tokenizer=get_tokenizer_attr()
    )
}


for k in keys(config[DIR_DSET]):    # Accommodate other OS
    # ic(k)
    k = f'{DIR_DSET}.{k}'
    val = get(config, k)
    if k[k.rfind('.')+1:] == 'dir_nm':
        set_(config, k, os.path.join(*val.split('/')))


if __name__ == '__main__':
    import json
    from data_path import *

    fl_nm = 'config.json'
    ic(config)
    print(config)
    open(fl_nm, 'a').close()  # Create file in OS
    with open(f'{PATH_BASE}/{DIR_PROJ}/{fl_nm}', 'w') as f:
        json.dump(config, f, indent=4)

