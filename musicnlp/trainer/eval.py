"""
Generate from trained reformer, no seed per `hash_seed`
"""
import os
import json
import datetime
from os.path import join as os_join
from typing import Tuple, Dict, Iterable, Any, Union
from collections import OrderedDict

import torch

from stefutil import *
from musicnlp.util import *
from musicnlp.vocab import VocabType, MusicVocabulary, MusicTokenizer
from musicnlp.preprocess import KeyFinder, MusicConverter, transform
from musicnlp.models import MyReformerModelWithLMHead, MyTransfoXLLMHeadModel
from musicnlp.trainer.wordpiece_tokenizer import load_trained_tokenizer as load_wordpiece_tokenizer


def load_trained(
        model_name: str = None, directory_name:  Union[str, Iterable[str]] = None,
        model_key: Tuple[str, str, str, str] = None, mode: str = 'full'
) -> Union[MyReformerModelWithLMHead, MyTransfoXLLMHeadModel]:
    ca.check_mismatch('Model Name', model_name, ['reformer', 'transfo-xl'])
    if not hasattr(load_trained, 'key2path'):
        load_trained.key2path = dict(
            full={
                # (model name, datasets, #epoch, comment) => path
                # pretty good generation
                ('reformer', 'P&M', '256-256ep', 'mid-pch'): ['2022-10-03_11-58-11_reformer', 'trained'],

                # channel mixup
                ('reformer', 'All', '5-16ep', 'mid-pch_1e-4'): ['2022-10-09_01-36-18_reformer', 'checkpoint-6850'],
                # w/ a loss slightly larger than the last one, generated not good
                ('reformer', 'All', '16-16ep', 'mid-pch_1e-4'): ['2022-10-09_01-36-18_reformer', 'trained'],

                # 1st try of proportional mixing, model w/ the best loss loaded in the end
                ('reformer', 'All', 'x-128ep', '1st-prop-mix'): ['2022-10-15_22-44-10_reformer', 'trained'],

                ('transf-xl', 'All', 'x-128ep', 'prop-mix'): ['2022-10-19_04-50-21_transf-xl', 'trained'],  # midi pitch
                # degree pitch, eval set no channel mixup
                ('transf-xl', 'All', '128-128ep', 'deg-pch_eval-no-mixup'): [
                    '2022-10-26_08-41-26_transf-xl', 'trained'],
                ('transf-xl', 'All', '128-128ep', 'with-crop'): ['2022-10-27_07-56-03_transf-xl', 'trained'],
                ('transf-xl', 'All', '256-256ep', 'with-crop_train-longer'): [
                    '2022-10-29_08-28-57_transf-xl', 'trained'],

                ('transf-xl', 'All', '128ep', 'no-mixup'): ['2022-11-11_18-04-07_transf-xl', 'trained'],
                ('transf-xl', 'All', '128ep', 'midi'): ['2022-11-14_13-04-30_transf-xl', 'trained'],

                ('transf-xl', 'All', '128ep', 'midi_no-wp'): ['2022-11-18_18-22-47_transf-xl', 'checkpoint-10863'],

                ('transf-xl', 'All', '128ep', 'midi_longer-seq'): ['2022-11-21_21-22-24_transf-xl', 'checkpoint-30348'],
                ('transf-xl', 'All', '128ep', 'degree_no-wp'): ['2022-11-24_01-18-17_transf-xl', 'checkpoint-7755'],
                ('transf-xl', 'All', '128ep', 'degree_no-wp_2'): ['2022-11-24_16-29-59_transf-xl', 'trained'],

                ('transf-xl', 'All', '128ep', 'no-wp_seg-len-512'): ['2022-11-27_13-03-40_transf-xl', 'trained'],

                ('transf-xl', 'All', '128ep', 'large-wp'): ['2022-11-28_15-52-20_transf-xl', 'trained']
            }
        )
    paths = [get_base_path(), u.model_dir]
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
    logger.info(f'Loaded {pl.i(cls.__qualname__)} with config {pl.fmt(model.config.to_dict())}')
    if model_name == 'transf-xl':
        model.config.pad_token_id = model.config.eos_token_id  # for open-end generation
    return model


class MusicGenerator:
    """
    Wraps a model for music generation
    """
    key2key_path_out = OrderedDict(dict(
        n_bar='#b',
        top_p='topp',
        top_k='topk',
        num_beams='#bm',
        repetition_penalty='rp',
        penalty_alpha='pa'
    ))

    def __init__(
            self, model: Union[MyReformerModelWithLMHead, MyTransfoXLLMHeadModel], mode: str = 'full',
            max_length: int = None, vocab: MusicVocabulary = None,
            wordpiece_tokenize: bool = True, tokenizer_args: Dict = None, augment_key: bool = False
    ):
        self.model = model
        if max_length:
            self.max_len = max_length
        else:
            if isinstance(model, MyReformerModelWithLMHead):
                self.max_len = model.config.max_position_embeddings
            else:  # transf xl
                self.max_len = model.config.max_length_
        tokenizer_args = dict(model_max_length=self.max_len) | tokenizer_args
        if wordpiece_tokenize:
            self.tokenizer = load_wordpiece_tokenizer(omit_eos=True, **tokenizer_args)
        else:
            self.tokenizer = MusicTokenizer(**tokenizer_args)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # TODO: logging still shows
        self.vocab = vocab or self.tokenizer.vocab
        self.pitch_kind = self.vocab.pitch_kind
        self.mc = MusicConverter(
            mode=mode, precision=self.tokenizer.precision, vocab_midi=self.vocab
        )
        self.augment_key = augment_key

        sr_vocab = vocab if self.pitch_kind == 'step' else MusicVocabulary(pitch_kind='step')
        self.sr = transform.SanitizeRare(vocab=sr_vocab, for_midi=self.pitch_kind == 'midi', return_as_list=True)
        self.tmp = None
        if self.pitch_kind == 'midi':
            self.tmp = transform.ToMidiPitch(vocab=sr_vocab, return_as_list=True)

        self.logger = get_logger('Music Generator')
        d_log = dict(model_max_length=self.max_len)
        self.logger.info(f'{pl.i(self.__class__.__qualname__)} initialized w/ {pl.i(d_log)}')

    @staticmethod
    def args2fnm(args: dict):
        out = args['strategy']
        for k, k_out in MusicGenerator.key2key_path_out.items():
            if k in args:
                out = f'{out}, {k_out}={args[k]}'
        return f'{{{out}}}'

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
        ins_key, pch_sft = (prompt_args.get(k, False) for k in ('insert_key', 'pitch_shift'))
        if ins_key is True:
            path = prompt_args.get('path', None)
            assert path, f'A path to a song must be provided to {pl.i("prompt_args")} to extract key ' \
                         f'when key is not already provided'
            ins_key = pt_sample(KeyFinder(path)(return_type='dict'))  # just sample a key for generation, TODO?
        if mode == 'unconditional':
            # TODO: sample the time signature and tempos?
            prompt = [self.vocab.meta2tok(VocabType.time_sig, (4, 4)), self.vocab.meta2tok(VocabType.tempo, 120)]
            # TODO: randomly sample a key?
            prompt = ' '.join([*prompt, self.vocab.start_of_bar])
        else:  # 'conditional'
            prompt, path = prompt_args.get('prompt', None), prompt_args.get('path', None)
            assert prompt is not None or path is not None, f'Expect either {pl.i("prompt")} or {pl.i("path")}'
            if prompt is None:
                prompt = self.mc.mxl2str(path, n_bar=prompt_args['n_bar'], insert_key=ins_key)

            prompt = self.sr(prompt)

            if self.pitch_kind == 'midi':
                prompt = self.tmp(prompt)
            elif pch_sft:
                ps = transform.PitchShift()
                prompt = ps(prompt)
            if isinstance(prompt, list):
                prompt = ' '.join(prompt)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        # inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])  # TODO: generation warning still shows
        args = dict(max_length=self.max_len)
        if strategy == 'greedy':
            assert (k in ['do_sample'] for k in generate_args)
            if 'do_sample' in generate_args:
                assert not generate_args['do_sample'], f'{pl.i("do_sample")} must be False for greedy generation'
            else:
                generate_args['do_sample'] = False
        elif strategy == 'sample':
            assert all(
                k in ['do_sample', 'top_k', 'top_p', 'temperature', 'repetition_penalty', 'penalty_alpha']
                for k in generate_args
            )
            if 'do_sample' in generate_args:
                assert generate_args['do_sample'], f'{pl.i("do_sample")} must be True for sample generation'
            else:
                generate_args['do_sample'] = True
            assert 'top_k' in generate_args or 'top_p' in generate_args, \
                f'Expect either {pl.i("top_k")} or {pl.i("top_p")} for sample generation'
        else:  # seems to repeat itself
            assert strategy == 'beam'
            generate_args = dict(num_beams=4, early_stopping=True) | generate_args
            assert (k in ['do_sample', 'num_beams', 'num_beam_groups', 'early_stopping'] for k in generate_args)
            assert generate_args['num_beams'] > 1, f'{pl.i("num_beams")} must >1 for beam-search generation'
        args |= generate_args
        prompt_colored = self.tokenizer.colorize(prompt)
        d_log = dict(mode=mode, strategy=strategy, args=generate_args | prompt_args, prompt=prompt_colored)
        self.logger.info(f'Generating with {pl.i(d_log)}')
        t = datetime.datetime.now()
        self.model.eval()
        output = self.model.generate(**inputs, **args)  # for now, generate one at a time
        self.logger.info(f'Model generation finished in {pl.i(fmt_delta(datetime.datetime.now() - t))}')
        assert len(output) == 1  # sanity check
        output = output[0]

        if truncate_to_sob:
            idxs_eob = torch.nonzero(output.eq(self.tokenizer.sob_token_id)).flatten().tolist()
            assert len(idxs_eob) > 0, f'No start of bar token found when {pl.i("truncate_to_sob")} enabled'
            output = output[:idxs_eob[-1]]  # truncate also that `sob_token`
        decoded = self.tokenizer.decode(output, skip_special_tokens=False)
        title = f'{save}-generated' if save else None
        score = self.mc.str2score(
            decoded, omit_eos=True, title=title, pitch_kind=self.pitch_kind, 
            check_duration_match=True
        )
        if save:
            # `makeNotations` disabled any clean-up by music21, intended to remove `tie`s added
            str_args = MusicGenerator.args2fnm(dict(strategy=strategy) | args | prompt_args)
            out_path = u.eval_path
            if save_dir:
                out_path = os_join(out_path, save_dir)
                os.makedirs(out_path, exist_ok=True)
            date = now(for_path=True)
            fnm = f'{date}_{title}_{str_args}'
            with open(os_join(out_path, f'{fnm}.json'), 'w') as f:
                d_log['prompt'] = prompt  # remove color
                json.dump(dict(meta=d_log, generation_args=args, generated=decoded), f, indent=4)
            try:
                # TODO: `makeNotation` False always breaks on GL
                score.write(fmt='mxl', fp=os_join(out_path, f'{fnm}.mxl'), makeNotation=False)
            except Exception as e:
                vocab = self.mc.vocabs.degree if self.augment_key else self.mc.vocabs.midi
                raise ValueError(f'Failed to render MXL from decoded output {vocab.colorize_tokens(decoded)}') from e
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

    md_k = md_nm, ds_nm, ep_nm, desc = 'transf-xl', 'All', '128ep', 'large-wp'
    mic(md_nm, ds_nm, ep_nm, desc)

    # pch_kd = 'midi'
    pch_kd = 'degree'
    tk_args = dict(pitch_kind=pch_kd)

    wp = True
    tk_args['fnm'] = '22-11-26_WordPiece-Tokenizer_{dnm=all}_{vsz=262144, n=178825, pch=d, aug-key=T}'

    md = 'full'
    mdl = load_trained(model_key=md_k, mode=md)
    sv_dir = f'{md_nm}_{ds_nm}_{ep_nm}_{desc}'
    # save_dir_ = 'reformer-base, 14/32ep'
    # mic(get_model_num_trainable_parameter(mdl))
    # step vocab for `MusicConverter::mxl2str`

    key_aug = True
    mg = MusicGenerator(model=mdl, mode=md, tokenizer_args=tk_args, augment_key=True, wordpiece_tokenize=wp)

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
        prompt_args = dict(path=path, n_bar=4, insert_key=True)
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

    def export_generated(batched: bool = True):
        pch_sft = True
        fnms = [
            # 'Careless Whisper, 4',
            # 'Ode to Joy',
            'Canon piano', 'Shape of You', 'Piano Sonata', '平凡之路', 'Merry Go Round of Life',
            # 'Merry Christmas',

            "Stayin' Alive", 'Señorita', 'Sugar', 'Something Just Like This', 'See You Again',

            'Für Elise', 'Moonlight', 'Symphony No.5', 'Flower Duet', 'The Marriage of Figaro', 'Serenade No. 13',
            'KV 448', 'William Tell',

            'My Heart Will Go On', 'Rolling in the Deep', 'Hallelujah'
        ]
        fnm2bar = {
            # 'Merry Go Round of Life': 4
            'Moonlight': 4
        }
        # gen_args = dict(top_k=16, top_p=0.75)  # this set up causes repetitions early on
        # gen_args = dict(top_k=32, top_p=0.95)
        # gen_args = dict(top_k=32, top_p=0.9)  # Kinda good for `All`
        # gen_args = dict(top_k=64, top_p=0.9)
        # gen_args = dict(top_k=32, top_p=0.75)  # Good w/ `P&M`, and 5-16 All
        # gen_args = dict(top_k=32, top_p=0.85)

        # gen_args = dict(top_k=32)
        # gen_args = dict(top_k=64, temperature=2.0)

        # gen_args = dict(top_p=0.75)
        # gen_args = dict(top_p=0.85)
        # gen_args = dict(top_p=0.85, repetition_penalty=1.2)  # penalty as in CTRL paper

        # gen_args = dict(top_k=32, penalty_alpha=0.3)
        # gen_args = dict(top_k=16, penalty_alpha=0.3)
        # gen_args = dict(top_k=8, penalty_alpha=0.5)
        gen_args = dict(top_k=8, penalty_alpha=0.6)  # Pretty good
        # gen_args = dict(top_k=16, penalty_alpha=0.6)
        # gen_args = dict(top_k=12, penalty_alpha=0.6)

        # n_bar = 4
        n_bar = 8
        for fnm in fnms:
            mic(fnm)
            path = music_util.get_my_example_songs(k=fnm, extracted=True, postfix='{md=f}')
            prompt = dict(path=path, n_bar=fnm2bar.get(fnm, n_bar), insert_key=key_aug, pitch_shift=pch_sft)

            def call():
                mg(
                    mode='conditional', strategy='sample', generate_args=gen_args, prompt_args=prompt,
                    save=fnm, save_dir=sv_dir
                )
            if batched:
                try:
                    call()
                except Exception as e:
                    print(f'Failed to generate {pl.i(fnm)} due to {e}')
            else:
                call()
    export_generated(batched=True)

    def eval_ikr():
        md_sz = 'debug'
    # eval_ikr()
