"""
Generate from trained reformer, no seed per `hash_seed`
"""
import os
import datetime
from typing import Dict, Iterable, Any, Union
from collections import OrderedDict

import torch
from transformers import ReformerModelWithLMHead

from musicnlp.util import *
from musicnlp.vocab.music_vocab import VocabType
from musicnlp.vocab import MusicTokenizer, MusicConverter
from musicnlp.preprocess import KeyFinder


def load_trained(model_name: str, directory_name:  Union[str, Iterable[str]]):
    paths = [PATH_BASE, DIR_PROJ, DIR_MDL, model_name]
    if isinstance(directory_name, str):
        paths.append(directory_name)
    else:
        paths.extend(directory_name)
    path = os.path.join(*paths)
    return ReformerModelWithLMHead.from_pretrained(path)


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

    def __init__(self, model: ReformerModelWithLMHead, deprecated: bool = False):
        self.model = model
        self.max_len = model.config.max_position_embeddings
        self.tokenizer = MusicTokenizer(model_max_length=self.max_len, deprecated=deprecated)
        self.vocab = self.tokenizer.vocab
        self.converter = MusicConverter()

        self.eval_path = os.path.join(PATH_BASE, DIR_PROJ, 'evaluations')
        os.makedirs(self.eval_path, exist_ok=True)

        self.logger = get_logger('Music Generator')

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

        generate_args = dict(num_beams=4) | (generate_args or dict())
        prompt_args: Dict[str, Any] = dict(n_bar=4, insert_key=False) | (prompt_args or dict())
        key = prompt_args['insert_key']
        if key is True:
            path = prompt_args.get('path', None)
            assert path, f'A path to a song must be provided to {logi("prompt_args")} to extract key ' \
                         f'when key is not already provided'
            key = sample(KeyFinder(path).find_key(return_type='dict'))  # just sample a key for generation, TODO?
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
            assert (k in ['do_sample', 'top_k', 'top_p'] for k in generate_args)
            if 'do_sample' in generate_args:
                assert generate_args['do_sample'], f'{logi("do_sample")} must be True for sample generation'
            else:
                generate_args['do_sample'] = True
            assert 'top_k' in generate_args or 'top_p' in generate_args, \
                f'Expect either {logi("top_k")} or {logi("top_p")} for sample generation'
        else:
            assert strategy == 'beam'
            assert (k in ['do_sample', 'num_beams', 'num_beam_groups'] for k in generate_args)
            assert generate_args['num_beams'] > 1, f'{logi("num_beams")} must >1 for beam-search generation'
        args |= generate_args
        ic(MusicGenerator.args2fnm(dict(strategy=strategy) | args | prompt_args))
        prompt_colored = ' '.join([self.tokenizer.vocab.colorize_token(tok) for tok in self.tokenizer.tokenize(prompt)])
        d_log = dict(mode=mode, strategy=strategy, args=generate_args | prompt_args, prompt=prompt_colored)
        self.logger.info(f'Generating with {log_dict(d_log)}')
        t = datetime.datetime.now()
        out = self.model.generate(**inputs, **args)  # TODO: debugging
        self.logger.info(f'Model generation finished in {logi(fmt_time(datetime.datetime.now() - t))}')
        ic(out, len(out))
        outputs = self.model.generate(**inputs, **args)[0]  # for now, generate one at a time

        if truncate_to_sob:
            idxs_eob = torch.nonzero(outputs.eq(self.tokenizer.sob_token_id)).flatten().tolist()
            assert len(idxs_eob) > 0, f'No start of bar token found when {logi("truncate_to_sob")} enabled'
            outputs = outputs[:idxs_eob[-1]]  # truncate also that `sob_token`
        decoded = self.tokenizer.decode(outputs, skip_special_tokens=False)
        title = f'{save}-generated' if save is not None else None
        score = self.converter.str2score(decoded, omit_eos=True, title=title)  # incase model can't finish generation
        if save:
            # `makeNotations` disabled any clean-up by music21, intended to remove `tie`s added
            str_args = MusicGenerator.args2fnm(args | prompt_args)
            out_path = self.eval_path
            ic(save_dir)
            if save_dir:
                out_path = os.path.join(save_dir, out_path)
            path = os.path.join(out_path, f'{title}, {str_args}, {now(for_path=True)}.mxl')
            score.write(fmt='mxl', fp=path, makeNotation=False)
        else:
            score.show()


if __name__ == '__main__':
    from icecream import ic

    import musicnlp.util.music as music_util

    # dir_nm = os.path.join('2022-04-01_09-40-48', 'trained')
    # dir_nm = os.path.join('2022-04-03_11-01-04', 'checkpoint-3712')
    # dir_nm = os.path.join('2022-04-11_00-26-05', 'trained')
    dir_nm = ['2022-04-16_16-08-03', 'checkpoint-4802']
    mdl = load_trained(model_name='reformer', directory_name=dir_nm)
    save_dir_ = 'reformer-base, 14/32ep'
    # ic(get_model_num_trainable_parameter(mdl))
    # mg = MusicGenerator(mdl)
    deprec = True
    mg = MusicGenerator(mdl, deprecated=deprec)

    def explore_generate_unconditional():
        # as in `CTRL` paper
        # still repeats itself, and notion of penalizing repetition with music generation?
        # gen_args = dict(repetition_penalty=1.2)
        # how to set hyperparameters?; smaller k for smaller vocab size
        # gen_args = dict(temperature=1, top_k=16)
        gen_args = dict(top_k=0, top_p=0.9)
        mg(mode='unconditional', strategy='sample', generate_args=gen_args)
    # explore_generate_unconditional()

    def explore_generate_conditional():
        fnm = 'Merry Go Round of Life'
        path = music_util.get_my_example_songs(k=fnm, extracted=True)
        # strat = 'greedy', None
        # strat, gen_args = 'sample', dict(top_k=32, top_p=0.9)
        strat, gen_args = 'beam', dict(num_beams=4, num_beam_groups=2)
        prompt_args = dict(path=path, n_bar=4, insert_key=not deprec)
        mg(
            mode='conditional', strategy=strat, generate_args=gen_args, prompt_args=prompt_args, save=fnm,
            save_dir=save_dir_
        )
    explore_generate_conditional()

    def check_why_tie_in_output():
        import music21 as m21
        incorrect_tie_fl = '/Users/stefanh/Desktop/incorrect tie.mxl'
        score: m21.stream.Score = m21.converter.parse(incorrect_tie_fl)
        for bar in list(score.parts)[0][m21.stream.Measure]:
            for e in bar:
                if isinstance(e, (m21.note.Note, m21.note.Rest)):
                    ic(e, e.tie)
    # check_why_tie_in_output()

    def export_generated():
        # fnms = ['Merry Go Round of Life', 'Shape of You']
        fnms = ['Canon piano', 'Shape of You']
        # gen_args = dict(top_k=16, top_p=0.75)  # this set up causes repetitions early on
        gen_args = dict(top_k=32, top_p=0.9)
        n_bar = 4
        for fnm in fnms:
            path = music_util.get_my_example_songs(k=fnm, extracted=True)
            prompt = dict(path=path, n_bar=n_bar)
            mg(
                mode='conditional', strategy='sample', generate_args=gen_args, prompt_args=prompt,
                save=fnm, save_dir=save_dir_
            )
    export_generated()
