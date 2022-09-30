"""
Generate from trained reformer, no seed per `hash_seed`
"""
import os
from os.path import join as os_join
import datetime
from typing import Tuple, Dict, Iterable, Any, Union
from collections import OrderedDict

import torch

from stefutil import *
from musicnlp.util import *
from musicnlp.vocab.music_vocab import VocabType
from musicnlp.vocab import MusicTokenizer
from musicnlp.preprocess import KeyFinder, MusicConverter
from musicnlp.models import MyReformerModelWithLMHead, MyTransfoXLLMHeadModel


def load_trained(
        model_name: str = None, directory_name:  Union[str, Iterable[str]] = None, model_key: Tuple[str, str] = None,
        mode: str = 'full'
) -> Union[MyReformerModelWithLMHead, MyTransfoXLLMHeadModel]:
    if not hasattr(load_trained, 'key2path'):
        load_trained.key2path = dict(
            melody={
                ('reformer', '14_32ep'): ['reformer', '2022-04-16_16-08-03', 'checkpoint-4802'],
                ('reformer', '20_64ep'): ['reformer', '2022-04-19_13-48-54', 'checkpoint-6860'],
                ('reformer', '2_16ep'): ['2022-06-23_01-11-20_reformer', 'checkpoint-5478'],
                ('transf-xl', '7_10ep'): ['2022-06-19_10-59-06_transf-xl', 'checkpoint-61341']
            },
            full={
                ('reformer', '32_32ep'): ['2022-08-07_05-12-51_reformer', 'trained']
            }
        )
    paths = [u.model_path]
    if mode == 'melody':  # different # of tokens in vocab
        raise NotImplementedError('Current Tokenizer don\'t support prior melody-only representation ')
    if model_key:
        model_name = model_key[0]
        paths.extend(load_trained.key2path[mode][model_key])
    else:
        if isinstance(directory_name, str):
            paths.append(directory_name)
        else:
            paths.extend(directory_name)
    path = os_join(*paths)
    cls = MyReformerModelWithLMHead if model_name == 'reformer' else MyTransfoXLLMHeadModel
    logger = get_logger('Load Trained')
    model = cls.from_pretrained(path)
    logger.info(f'Loaded {logi(cls.__qualname__)} with config {log_dict_pg(model.config.to_dict())}')
    return model


class MusicGenerator:
    """
    Wraps a model for music generation
    """
    key2key_path_out = OrderedDict(dict(
        top_p='topp',
        top_k='topk',
        num_beams='#beam',
        n_bar='#bar'
    ))

    def __init__(
            self, model: Union[MyReformerModelWithLMHead, MyTransfoXLLMHeadModel], mode: str = 'full',
            max_length: int = None
    ):
        self.model = model
        if max_length:
            self.max_len = max_length
        else:
            if isinstance(model, MyReformerModelWithLMHead):
                self.max_len = model.config.max_position_embeddings
            else:  # transf xl
                self.max_len = model.config.max_length_
        tk_args = dict(model_max_length=self.max_len)
        self.tokenizer = MusicTokenizer(**tk_args)
        self.vocab = self.tokenizer.vocab
        self.converter = MusicConverter(mode=mode, tokenizer_kw=tk_args)

        self.logger = get_logger('Music Generator')
        d_log = dict(model_max_length=self.max_len)
        self.logger.info(f'{logi(self.__class__.__qualname__)} initialized with max_len={logi(d_log)}')

    @staticmethod
    def args2fnm(args: dict):
        out = args['strategy']
        for k, k_out in MusicGenerator.key2key_path_out.items():
            if k in args:
                out = f'{out}, {k_out}={args[k]}'
        return out

    def __call__(
            self, mode: str, strategy: str, to_score: bool = False,
            generate_args: dict = None, prompt_args: dict = None,
            truncate_to_sob: bool = True, save: Union[bool, str] = False,
            save_dir: str = None
    ):
        """
        :param mode: One of [`conditional`, `unconditional`]
            If `conditional`, expect either `prompt` as string or path to a `musicnlp` extracted MXL file
                in `condition_args`
        :param strategy: One of [`greedy`, `sample`, `beam`]
            If `sample`, expect `topk`, `temperature`, `top_p` in `generate_args`
        :param prompt_args: Specifies prompt construction for conditional generation
            If 'insert_key', a key will be inserted into the prompt if needed
        :param truncate_to_sob: If True, truncate the generated tokens such that the generated ends in terms of bar
            Intended for converting to MXL
        :param save: If false, the score is shown
            Otherwise, should be a filename string, the generated score is saved to an MXL file
        """
        ca(generation_mode=mode, generation_strategy=strategy)

        generate_args = (generate_args or dict())
        prompt_args: Dict[str, Any] = dict(n_bar=4, insert_key=False) | (prompt_args or dict())
        key = prompt_args['insert_key']
        if key is True:
            path = prompt_args.get('path', None)
            assert path, f'A path to a song must be provided to {logi("prompt_args")} to extract key ' \
                         f'when key is not already provided'
            key = pt_sample(KeyFinder(path).find_key(return_type='dict'))  # just sample a key for generation, TODO?
        if mode == 'unconditional':
            # TODO: sample the time signature and tempos?
            prompt = [self.vocab.uncompact(VocabType.time_sig, (4, 4)), self.vocab.uncompact(VocabType.tempo, 120)]
            if key:
                prompt.append(key)
            prompt = ' '.join([*prompt, self.vocab.start_of_bar])
        else:  # 'conditional'
            prompt, path = prompt_args.get('prompt', None), prompt_args.get('path', None)
            assert prompt is not None or path is not None, f'Expect either {logi("prompt")} or {logi("path")}'
            if prompt is None:
                prompt = self.converter.mxl2str(path, n_bar=prompt_args['n_bar'], insert_key=key)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        args = dict(max_length=self.max_len)
        if strategy == 'greedy':
            assert (k in ['do_sample'] for k in generate_args)
            if 'do_sample' in generate_args:
                assert not generate_args['do_sample'], f'{logi("do_sample")} must be False for greedy generation'
            else:
                generate_args['do_sample'] = False
        elif strategy == 'sample':
            assert all(k in ['do_sample', 'top_k', 'top_p', 'temperature'] for k in generate_args)
            if 'do_sample' in generate_args:
                assert generate_args['do_sample'], f'{logi("do_sample")} must be True for sample generation'
            else:
                generate_args['do_sample'] = True
            assert 'top_k' in generate_args or 'top_p' in generate_args, \
                f'Expect either {logi("top_k")} or {logi("top_p")} for sample generation'
        else:  # seems to repeat itself
            assert strategy == 'beam'
            generate_args = dict(num_beams=4, early_stopping=True) | generate_args
            assert (k in ['do_sample', 'num_beams', 'num_beam_groups', 'early_stopping'] for k in generate_args)
            assert generate_args['num_beams'] > 1, f'{logi("num_beams")} must >1 for beam-search generation'
        args |= generate_args
        prompt_colored = ' '.join([self.tokenizer.vocab.colorize_token(tok) for tok in self.tokenizer.tokenize(prompt)])
        d_log = dict(mode=mode, strategy=strategy, args=generate_args | prompt_args, prompt=prompt_colored)
        self.logger.info(f'Generating with {log_dict(d_log)}')
        t = datetime.datetime.now()
        self.model.eval()
        output = self.model.generate(**inputs, **args)  # for now, generate one at a time
        self.logger.info(f'Model generation finished in {logi(fmt_delta(datetime.datetime.now() - t))}')
        assert len(output) == 1  # sanity check
        output = output[0]

        if truncate_to_sob:
            idxs_eob = torch.nonzero(output.eq(self.tokenizer.sob_token_id)).flatten().tolist()
            assert len(idxs_eob) > 0, f'No start of bar token found when {logi("truncate_to_sob")} enabled'
            output = output[:idxs_eob[-1]]  # truncate also that `sob_token`
        decoded = self.tokenizer.decode(output, skip_special_tokens=False)
        # mic(decoded)
        title = f'{save}-generated' if save else None
        score = self.converter.str2score(decoded, omit_eos=True, title=title)  # incase model can't finish generation
        if save:
            # `makeNotations` disabled any clean-up by music21, intended to remove `tie`s added
            str_args = MusicGenerator.args2fnm(dict(strategy=strategy) | args | prompt_args)
            out_path = u.eval_path
            if save_dir:
                out_path = os_join(out_path, save_dir)
                os.makedirs(out_path, exist_ok=True)
            path = os_join(out_path, f'{now(for_path=True)}, {title}, {str_args}.mxl')
            score.write(fmt='mxl', fp=path, makeNotation=False)
        else:
            score.show()


def get_performance(model):
    """
    Get NTP IKR on the eval set
    """
    # TODO
    pass


if __name__ == '__main__':
    import musicnlp.util.music as music_util

    # md_k = 'transf-xl', '7_10ep'
    md_k = 'reformer', '32_32ep'
    md = 'full'
    mdl = load_trained(model_key=md_k, mode=md)
    sv_dir = f'{md_k[0]}, {md_k[1]}'
    # save_dir_ = 'reformer-base, 14/32ep'
    # mic(get_model_num_trainable_parameter(mdl))
    mg = MusicGenerator(model=mdl, mode=md)

    # key_aug = False
    key_aug = True

    def explore_generate_unconditional():
        # as in `CTRL` paper
        # still repeats itself, and notion of penalizing repetition with music generation?
        # gen_args = dict(repetition_penalty=1.2)
        # how to set hyperparameters?; smaller k for smaller vocab size
        # gen_args = dict(temperature=1, top_k=16)
        gen_args = dict(top_k=32, top_p=0.9)
        mg(mode='unconditional', strategy='sample', generate_args=gen_args)
    # explore_generate_unconditional()

    def explore_generate_conditional():
        # fnm = 'Merry Go Round of Life'
        fnm = 'Canon piano'
        path = music_util.get_my_example_songs(k=fnm, extracted=True)
        # strat = 'greedy', None
        # strat, gen_args = 'sample', dict(top_k=32, top_p=0.9)
        strat, gen_args = 'sample', dict(top_k=64, top_p=0.9, temperature=1)
        # strat, gen_args = 'beam', dict(num_beams=4, num_beam_groups=2)
        prompt_args = dict(path=path, n_bar=4, insert_key=key_aug)
        mg(
            mode='conditional', strategy=strat, generate_args=gen_args, prompt_args=prompt_args, save=fnm,
            save_dir=sv_dir
        )
    # explore_generate_conditional()

    def check_why_tie_in_output():
        import music21 as m21
        incorrect_tie_fl = '/Users/stefanh/Desktop/incorrect tie.mxl'
        score: m21.stream.Score = m21.converter.parse(incorrect_tie_fl)
        for bar in list(score.parts)[0][m21.stream.Measure]:
            for e in bar:
                if isinstance(e, (m21.note.Note, m21.note.Rest)):
                    mic(e, e.tie)
    # check_why_tie_in_output()

    def export_generated():
        fnms = ['Merry Go Round of Life', 'Canon piano', 'Shape of You', 'Merry Christmas']
        # fnms = ['Merry Christmas']
        # fnms = ['Faded', 'Piano Sonata', 'Merry Christmas']
        # gen_args = dict(top_k=16, top_p=0.75)  # this set up causes repetitions early on
        # gen_args = dict(top_k=32, top_p=0.95)
        # gen_args = dict(top_k=32, top_p=0.9)
        # gen_args = dict(top_k=64, top_p=0.9)
        gen_args = dict(top_k=32, top_p=0.75)
        n_bar = 4
        for fnm in fnms:
            path = music_util.get_my_example_songs(k=fnm, extracted=True, postfix='full')
            prompt = dict(path=path, n_bar=n_bar, insert_key=key_aug)
            mg(
                mode='conditional', strategy='sample', generate_args=gen_args, prompt_args=prompt,
                save=fnm, save_dir=sv_dir
            )
    export_generated()

    def eval_ikr():
        md_sz = 'debug'
    # eval_ikr()
