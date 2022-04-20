from typing import List

from musicnlp.util.util import *


class CheckArg:
    """
    Raise errors when common arguments don't match the expected values
    """
    extraction_export_types = ['mxl', 'str', 'id', 'str_join', 'visualize']
    music_file_formats = ['mxl', 'midi']
    key_type = ['list', 'enum', 'dict']

    logging_strategy = ['epoch', 'steps', 'no']
    logging_mode = ['train', 'eval']

    generation_modes = ['conditional', 'unconditional']
    generation_strategies = ['greedy', 'sample', 'beam']

    plot_orients = ['v', 'h', 'vertical', 'horizontal']

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

    @staticmethod
    def check_key_type(key_type: str):
        CheckArg.check_mismatch('Keys Output Type', key_type, CheckArg.key_type)

    @staticmethod
    def check_logging_strategy(ls: str):
        CheckArg.check_mismatch('Train Logging Strategy', ls, CheckArg.logging_strategy)

    @staticmethod
    def check_logging_mode(mode: str):
        CheckArg.check_mismatch('Train Logging Mode', mode, CheckArg.logging_mode)

    @staticmethod
    def check_generation_mode(mode: str):
        CheckArg.check_mismatch('Music Generation Mode', mode, CheckArg.generation_modes)

    @staticmethod
    def check_generation_strategy(strategy: str):
        CheckArg.check_mismatch('Music Generation Strategy', strategy, CheckArg.generation_strategies)

    @staticmethod
    def check_plot_orient(orient: str):
        CheckArg.check_mismatch('Bar Plot Orientation', orient, CheckArg.plot_orients)

    def __init__(self):
        self.d_name2func = dict(
            exp=CheckArg.check_extraction_export_type,
            fmt=CheckArg.check_music_file_format,
            key_type=CheckArg.check_key_type,
            logging_strategy=CheckArg.check_logging_strategy,
            logging_mode=CheckArg.check_logging_mode,
            generation_mode=CheckArg.check_generation_mode,
            generation_strategy=CheckArg.check_generation_strategy,
            orient=CheckArg.check_plot_orient
        )

    def __call__(self, **kwargs):
        for k in kwargs:
            self.d_name2func[k](kwargs[k])


ca = CheckArg()


if __name__ == '__main__':
    exp_ = 'id'
    ca(exp=exp_)
