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
    def down_single(mode: str = None, dataset_name: str = None):
        fnm = f'md={mode}, dnm={dataset_name}'
        out_path = os_join(music_util.get_processed_path(), 'hf', f'{fnm}.zip')
        # mic(out_path, os.path.dirname())
        download_n_unzip(url=HF_DSETS2URL[mode, dataset_name], download_output_path=out_path)
    # down_single(mode='melody', dataset_name='MAESTRO')
    # down_single(mode='melody', dataset_name='POP909')

    def download_datasets():
        for mode, dataset_name in HF_DSETS2URL:
            down_single(mode=mode, dataset_name=dataset_name)
    download_datasets()

    def download_tokenizer():
        fnm = 'tokenizer, {md=full, dnm=all}'
        out_path = os_join(u.tokenizer_path, f'{fnm}.zip')
        download_n_unzip(url=TOKENIZER_URL, download_output_path=out_path)
    download_tokenizer()
