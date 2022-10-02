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


DROPBOX_LMD_FNM = '22-05-27_LMD, all.zip'

CONVERTED_FILES2URL = {
    # (`conversion backend`, `dataset name`) => Google Drive link url
    # a folder containing converted files from both MuseScore & Logic Pro
    # this one from UMich GDrive, some problem w/ gdown and large files
    # ('All', 'LMD'): 'https://drive.google.com/uc?id=1CyfKiVX83YdS4p7_4npk2xbDVJ68L0tg',
    ('MuseScore', 'LMD'): [
        'https://drive.google.com/uc?id=1-ISc2u6Sxvs3LES4byx0KcNGGVYDZnxV',
        'https://drive.google.com/uc?id=1LdXavGJnCeoKn2ZnkbbpYTbdwpNTjrLe',
        'https://drive.google.com/uc?id=11ci6rw6NqdoNEDgKIcTN_XZxRgG-Ie3K'
    ],
    ('Logic Pro', 'LMD'): 'https://drive.google.com/uc?id=1ZAxpnu9md7FY-5Cq9I8deAXWbOY0MZBs',
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


def download_n_unzip(url: str = None, download_output_path: str = None, extract_path: str = None):
    if not os.path.exists(download_output_path):
        logger.info(f'Downloading file from {logi(url)} to {logi(download_output_path)}... ')
        gdown.download(url, download_output_path, quiet=False)

    ext_pa = extract_path or os.path.dirname(download_output_path)
    logger.info(f'Unzipping downloaded file {logi(download_output_path)} to {logi(ext_pa)}... ')
    with ZipFile(download_output_path, 'r') as zf:
        zf.extractall(ext_pa)


if __name__ == '__main__':
    conv_base = os_join(get_output_base(), u.dset_dir, 'converted-debug')
    hf_base = os_join(music_util.get_processed_path(), 'hf')
    mic(conv_base, hf_base)
    os.makedirs(conv_base, exist_ok=True)
    os.makedirs(hf_base, exist_ok=True)

    def down_from_dropbox():
        # TODO: no longer needed since no `dropbox` python package in linux
        import dropbox
        import json
        with open(os_join(u.proj_path, 'auth', 'dropbox-access-token.json')) as f:
            token = json.load(f)['access-token']
        dbx = dropbox.Dropbox(token)
        # for entry in dbx.files_list_folder('').entries:
        #     mic(entry.name, entry)
        mic(dbx)
        with open(os_join(conv_base, DROPBOX_LMD_FNM), 'wb') as f:
            metadata, result = dbx.files_download(path=f'/{DROPBOX_LMD_FNM}')
            mic(metadata, type(result))
            f.write(result.content)
    # down_from_dropbox()

    def down_converted_single(back_end: str = None, dataset_name: str = None):
        fnm = f'Converted_{{be={back_end}, dnm={dataset_name}}}'
        url = CONVERTED_FILES2URL[(back_end, dataset_name)]
        if isinstance(url, list):
            assert back_end == 'MuseScore' and dataset_name == 'LMD'
            dnm = 'LMD, MS'
            for i, url_ in enumerate(url, start=1):
                out_path = os_join(conv_base, f'{fnm}_split {i}.zip')
                ext_path = os_join(conv_base, dnm)
                download_n_unzip(url=url_, download_output_path=out_path, extract_path=ext_path)
        else:
            out_path = os_join(conv_base, f'{fnm}.zip')
            download_n_unzip(url, out_path)
    # down_converted_single(back_end='MuseScore', dataset_name='POP909')
    # down_converted_single(back_end='MuseScore', dataset_name='MAESTRO')
    down_converted_single(back_end='MuseScore', dataset_name='LMD')
    down_converted_single(back_end='Logic Pro', dataset_name='LMD')

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
