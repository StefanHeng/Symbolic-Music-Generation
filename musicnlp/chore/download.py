"""
Download datasets & trained tokenizer for CLM training
"""

import os
from os.path import join as os_join
from zipfile import ZipFile

import gdown

from stefutil import *
from musicnlp.util import *
import musicnlp.util.music as music_util


logger = get_logger('Download')

CONVERTED_FILES2URL = {
    # (`conversion backend`, `dataset name`) => Google Drive link url
    # a folder containing converted files from both MuseScore & Logic Pro
    ('All', 'LMD'): 'https://drive.google.com/uc?id=1CyfKiVX83YdS4p7_4npk2xbDVJ68L0tg',
    ('MuseScore', 'MAESTRO'): 'https://drive.google.com/uc?id=1fzmfS65BN84O_bF1v8dN2uFlrrpOzYaZ',
    ('MuseScore', 'POP909'): 'https://drive.google.com/uc?id=1XobTD6x88PIEKfrZ6IAzXjMaZmBZ0XqR'
}

HF_DSETS2URL = {
    # (`mode`, `dataset name`) => Google Drive link url
    ('full', 'LMD'): 'https://drive.google.com/uc?id=16qDj2SJ8CoT4Tqacc3OZfsVZ6_6CDs1s',
    ('full', 'MAESTRO'): 'https://drive.google.com/uc?id=1UaXtvqloFojNc1RnZ8ZqqqeKuSAbCjOC',
    ('full', 'POP909'): 'https://drive.google.com/uc?id=1dSxBi8Z1If-HuiHP9eWaRQAjYiRUPgnN',
    ('melody', 'LMD'): 'https://drive.google.com/uc?id=1l5v_KN3-d-i7lP0Xo-Ifj1ZEJbYCwUbO',
    ('melody', 'MAESTRO'): 'https://drive.google.com/uc?id=1oiujQaeMUnd2-PmO7KIIsppVRo_eZtXz',
    ('melody', 'POP909'): 'https://drive.google.com/uc?id=1F07h0JGTSYZSpzrGm9wP1pA2tB-6phsL'
}

TOKENIZER_URL = 'https://drive.google.com/uc?id=1rbQccozpAMjRWkjtKConka_DkCusxZsF'


def download_n_unzip(url: str = None, download_output_path: str = None):
    if not os.path.exists(download_output_path):
        logger.info(f'Downloading file from {logi(url)} to {logi(download_output_path)}... ')
        gdown.download(url, download_output_path, quiet=False)

    logger.info(f'Unzipping downloaded file {logi(download_output_path)}... ')
    with ZipFile(download_output_path, 'r') as zf:
        zf.extractall(os.path.dirname(download_output_path))


if __name__ == '__main__':
    conv_base = os_join(get_output_base(), u.dset_dir, 'converted')
    hf_base = os_join(music_util.get_processed_path(), 'hf')
    mic(conv_base, hf_base)
    os.makedirs(conv_base, exist_ok=True)
    os.makedirs(hf_base, exist_ok=True)

    def down_converted_single(back_end: str = None, dataset_name: str = None):
        fnm = f'Converted_{{be={back_end}, dnm={dataset_name}}}'
        out_path = os_join(conv_base, f'{fnm}.zip')
        download_n_unzip(url=CONVERTED_FILES2URL[(back_end, dataset_name)], download_output_path=out_path)
    # down_converted_single(back_end='MuseScore', dataset_name='POP909')
    # down_converted_single(back_end='MuseScore', dataset_name='MAESTRO')
    # down_converted_single(back_end='All', dataset_name='LMD')

    def move_converted_lmd():
        """
        Cos unzipping LMD results in a folder, not exactly the path read in by MusicExtractor
        TODO: looks like not the case hence not needed...
        """
        import shutil
        fd_nm = ''
        fd_path = os_join(u.dset_path, 'converted', fd_nm)
        for p in os.listdir(fd_path):
            path_new = os_join(u.dset_path, 'converted', p)
            path_old = os_join(fd_path, p)
            shutil.move(src=path_old, dst=path_new)
    # move_converted_lmd()

    def down_hf_single(mode: str = None, dataset_name: str = None):
        fnm = f'HF_{{md={mode}, dnm={dataset_name}}}'
        out_path = os_join(hf_base, f'{fnm}.zip')
        # mic(out_path, os.path.dirname())
        download_n_unzip(url=HF_DSETS2URL[mode, dataset_name], download_output_path=out_path)
    # down_single(mode='melody', dataset_name='MAESTRO')
    # down_single(mode='melody', dataset_name='POP909')

    def download_hf_datasets():
        for mode, dataset_name in HF_DSETS2URL:
            down_hf_single(mode=mode, dataset_name=dataset_name)
    # download_hf_datasets()

    def download_tokenizer():
        fnm = 'tokenizer, {md=full, dnm=all}'
        out_path = os_join(u.tokenizer_path, f'{fnm}.zip')
        download_n_unzip(url=TOKENIZER_URL, download_output_path=out_path)
    # download_tokenizer()
