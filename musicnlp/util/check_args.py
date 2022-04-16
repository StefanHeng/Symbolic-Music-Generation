from typing import List

from musicnlp.util.util import *


class CheckArg:
    """
    Raise errors when common arguments don't match the expected values
    """
    extraction_export_types = ['mxl', 'str', 'id', 'str_join', 'visualize']
    music_file_formats = ['mxl', 'midi']

    @staticmethod
    def check_mismatch(arg_type: str, arg_value: str, expected_values: List[str]):
        if arg_value not in expected_values:
            raise ValueError(f'Unexpected {logi(arg_type)}: '
                             f'expect one of {logi(expected_values)}, got {logi(arg_value)}')

    @staticmethod
    def check_extraction_export_type(exp: str):
        CheckArg.check_mismatch('Extraction Export Type', exp, CheckArg.extraction_export_types)

    @staticmethod
    def check_music_file_format(fmt: str):
        CheckArg.check_mismatch('Music File Format', fmt, CheckArg.music_file_formats)

    def __init__(self):
        self.d_name2func = dict(
            exp=CheckArg.check_extraction_export_type,
            fmt=CheckArg.check_music_file_format
        )

    def __call__(self, **kwargs):
        for k in kwargs:
            self.d_name2func[k](kwargs[k])


ca = CheckArg()


if __name__ == '__main__':
    exp_ = 'id'
    ca(exp=exp_)
