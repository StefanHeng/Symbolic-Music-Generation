import os
import pickle
import logging
from os.path import join as os_join
from typing import Union
from fractions import Fraction

import colorama

from stefutil import *
from musicnlp.util.project_paths import BASE_PATH, PROJ_DIR, PKG_NM, DSET_DIR, MODEL_DIR


__all__ = [
    'sconfig', 'u', 'save_fig',
    'on_great_lakes', 'get_output_base',
    'serialize_frac', 'read_pickle'
]


sconfig = StefConfig(config_file=os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json')).__call__
u = StefUtil(
    base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR
)
u.tokenizer_path = os_join(u.base_path, u.proj_dir, 'tokenizers')
os.makedirs(u.tokenizer_path, exist_ok=True)
save_fig = u.save_fig

for d in sconfig('check-arg'):
    ca.cache_mismatch(**d)


def on_great_lakes():
    return 'arc-ts' in get_hostname()


def get_output_base(gl_account_name: str = 'mihalcea'):
    # For remote machines, save heavy-duty data somewhere else to save `/home` disk space
    if on_great_lakes():  # Great Lakes, see https://arc.umich.edu/greatlakes/user-guide/
        # `0` picked arbitrarily among [`0`, `1`]
        pa = os_join('/scratch', f'{gl_account_name}_root', f'{gl_account_name}0', 'stefanhg', stem(BASE_PATH))
        os.makedirs(pa, exist_ok=True)
        return pa
    else:
        return BASE_PATH


def serialize_frac(num: Union[Fraction, float]) -> Union[str, float]:
    return f'{num.numerator}/{num.denominator}' if isinstance(num, Fraction) else num


def read_pickle(fnm):
    objects = []
    with (open(fnm, 'rb')) as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    return objects


if __name__ == '__main__':
    # mic(config('Melody-Extraction.tokenizer'))

    def check_compress():
        arr = [
            202, 202, 202, 202, 203, 203, 203, 203, 202, 202, 202, 202, 203,
            203, 203, 203, 202, 202, 202, 202, 203, 203, 203, 203
        ]
        mic(compress(arr))
    # check_compress()

    # mic(quarter_len2fraction(1.25), quarter_len2fraction(0.875))

    # mic(hex2rgb('#E5C0FB'))

    def check_logging():
        mic(colorama.Fore.YELLOW)
        mic(now())
        mic()
        print('normal str')

        logger = logging.getLogger('Test Logger')
        logger.setLevel(logging.DEBUG)

        import sys
        ch = logging.StreamHandler(stream=sys.stdout)  # For my own coloring
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(MyFormatter())
        logger.addHandler(ch)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        res = colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL
        msg = colorama.Fore.YELLOW + 'my msg' + res
        logger.critical(msg)
    # check_logging()

    def check_group():
        lst = list(range(6))
        mic(lst, list(group_n(lst, 3)))
    # check_group()

    st = '/Users/stefanh/Documents/UMich/Research/Music with ' \
         'NLP/datasets/MXL-eg_out/Alpentrio Tirol - Alpentrio Hitmix: ' \
         'Alpentrio-Medley   Hast a bisserl Zeit fur mi   Tepperter Bua   Hallo kleine ' \
         'Traumfrau   Vergiss die Liebe nicht   Ich freu\' mich schon auf dich   Ich ' \
         'hab was ganz lieb\'s traumt von dir   Geheimnis der Joha... - v0.mxl'
    # mic(clean_whitespace(st))
