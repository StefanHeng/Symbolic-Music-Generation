import os
import pickle
import logging
from os.path import join as os_join
from typing import Union
from fractions import Fraction

import colorama

from stefutil import *
from musicnlp.util.data_path import BASE_PATH, PROJ_DIR, PKG_NM, DSET_DIR, MODEL_DIR


__all__ = ['sconfig', 'u', 'save_fig', 'serialize_frac', 'read_pickle']


sconfig = StefConfig(config_file=os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json')).__call__
u = StefUtil(
    base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR
)
u.tokenizer_path = os_join(u.base_path, u.proj_dir, 'tokenizers')
os.makedirs(u.tokenizer_path, exist_ok=True)
save_fig = u.save_fig

for d in sconfig('check-arg'):
    ca.cache_mismatch(**d)


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
    from icecream import ic

    # ic(config('Melody-Extraction.tokenizer'))

    def check_compress():
        arr = [
            202, 202, 202, 202, 203, 203, 203, 203, 202, 202, 202, 202, 203,
            203, 203, 203, 202, 202, 202, 202, 203, 203, 203, 203
        ]
        ic(compress(arr))
    # check_compress()

    # ic(quarter_len2fraction(1.25), quarter_len2fraction(0.875))

    # ic(hex2rgb('#E5C0FB'))

    def check_logging():
        ic(colorama.Fore.YELLOW)
        ic(now())
        ic()
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
        ic(lst, list(group_n(lst, 3)))
    # check_group()

    st = '/Users/stefanh/Documents/UMich/Research/Music with ' \
         'NLP/datasets/MXL-eg_out/Alpentrio Tirol - Alpentrio Hitmix: ' \
         'Alpentrio-Medley   Hast a bisserl Zeit fur mi   Tepperter Bua   Hallo kleine ' \
         'Traumfrau   Vergiss die Liebe nicht   Ich freu\' mich schon auf dich   Ich ' \
         'hab was ganz lieb\'s traumt von dir   Geheimnis der Joha... - v0.mxl'
    # ic(clean_whitespace(st))
