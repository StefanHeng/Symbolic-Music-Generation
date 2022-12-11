"""
Generate from trained reformer, no seed per `hash_seed`
"""

import os
import re
import json
import glob
import random
import datetime
import traceback
from os.path import join as os_join
from typing import List, Tuple, Dict, Iterable, Any, Union
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

                ('transf-xl', 'All', '128ep', 'large-wp'): ['2022-11-28_15-52-20_transf-xl', 'trained'],
                ('transf-xl', 'All', '128ep', 'no-ch-mix'): ['2022-11-30_20-00-10_transf-xl', 'trained'],

                ('transf-xl', 'All', '128ep', 'small_long-seq'): ['2022-12-04_16-03-03_transf-xl', 'trained'],
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
        diversity_penalty='dp',
        repetition_penalty='rp',
        temperature='t',
        penalty_alpha='pa',
        typical_p='typp'
    ))

    def __init__(
            self, model: Union[MyReformerModelWithLMHead, MyTransfoXLLMHeadModel], mode: str = 'full',
            max_length: int = None, vocab: MusicVocabulary = None,
            wordpiece_tokenize: bool = True, tokenizer_args: Dict = None, augment_key: bool = False,
            pick_key: str = 'sample'
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

        ca.check_mismatch('Select Key Scheme', pick_key, ['sample', 'max', 'first-2'])
        self.pick_key = pick_key

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
        out = args['strategy'][:4]
        for k, k_out in MusicGenerator.key2key_path_out.items():
            if k in args:
                out = f'{out}, {k_out}={args[k]}'
        return f'{{{out}}}'

    def _truncate_last_bar(self, ids: torch.Tensor) -> List[int]:
        assert ids.dim() == 1
        idxs_eob = torch.nonzero(ids.eq(self.tokenizer.sob_token_id)).flatten().tolist()
        assert len(idxs_eob) > 0, f'No start of bar token found when {pl.i("truncate_to_sob")} enabled'
        return ids[:idxs_eob[-1]].tolist()

    def __call__(
            self, mode: str, strategy: str, to_score: bool = False,
            generate_args: dict = None, prompt_args: dict = None,
            truncate_last_bar: bool = True, save: Union[bool, str] = False,
            save_dir: str = None
    ):
        """
        :param mode: One of [`conditional`, `unconditional`]
            If `conditional`, expect either `prompt` as string or path to a `musicnlp` extracted MXL file
                in `condition_args`
        :param strategy: One of [`greedy`, `sample`, `beam`]
            If `sample`, expect `topk`, `temperature`, `top_p` in `generate_args`
        :param prompt_args: Specifies prompt construction for generation
            For conditional generation, if 'insert_key', a key will be inserted into the prompt if needed
        :param truncate_last_bar: If True, truncate the generated tokens such that the generated ends in terms of bar
            Intended for converting to MXL
        :param save: If false, the score is shown
            Otherwise, should be a filename string, the generated score is saved to an MXL file
        """
        ca(generation_mode=mode, generation_strategy=strategy)
        ca.check_mismatch('Generation Mode', mode, ['conditional', 'unconditional'])

        generate_args = (generate_args or dict())
        prompt_args: Dict[str, Any] = dict(n_bar=4, insert_key=False) | (prompt_args or dict())
        ins_key, pch_sft = (prompt_args.get(k, False) for k in ('insert_key', 'pitch_shift'))
        if mode == 'unconditional':
            # TODO: sample the time signature and tempos?
            ts = prompt_args.get('time_signature', (4, 4))
            tp = prompt_args.get('tempo', 120)
            key = prompt_args.get('key', 'CMajor')
            prompt = [self.vocab.meta2tok(VocabType.time_sig, ts), self.vocab.meta2tok(VocabType.tempo, tp)]
            if ins_key:
                prompt += self.vocab(key)
            # TODO: randomly sample a key?
            prompt = ' '.join([*prompt, self.vocab.start_of_bar])
        else:  # 'conditional'
            if ins_key:
                path = prompt_args.get('path', None)
                assert path, f'A path to a song must be provided to {pl.i("prompt_args")} to extract key ' \
                             f'when key is not already provided'
                keys: Dict[str, float] = KeyFinder(path)(return_type='dict')
                if self.pick_key in ['sample', 'first-2']:
                    if self.pick_key == 'first-2':
                        _keys = sorted(keys.keys(), key=keys.get)[-2:]  # 2 highest-confidence keys
                        keys = {k: keys[k] for k in _keys}
                    ins_key = pt_sample(keys)
                else:  # `max`
                    ins_key = max(keys, key=keys.get)

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
        args = dict(max_length=self.max_len, early_stopping=True)

        accepted_keys_sample = ['do_sample', 'top_k', 'top_p', 'typical_p', 'temperature', 'repetition_penalty']
        if strategy == 'greedy':
            assert (k in ['do_sample'] for k in generate_args)
            if 'do_sample' in generate_args:
                assert not generate_args['do_sample'], f'{pl.i("do_sample")} must be False for greedy generation'
            else:
                generate_args['do_sample'] = False
        elif strategy == 'sample':
            assert all(k in accepted_keys_sample for k in generate_args)
            if 'do_sample' in generate_args:
                assert generate_args['do_sample'], f'{pl.i("do_sample")} must be True for sample generation'
            else:
                generate_args['do_sample'] = True
            assert 'top_k' in generate_args or 'top_p' in generate_args, \
                f'Expect either {pl.i("top_k")} or {pl.i("top_p")} for sample generation'
        elif strategy == 'contrastive':
            assert all(k in ['do_sample', 'top_k', 'penalty_alpha'] for k in generate_args)

            if 'do_sample' in generate_args:
                assert not generate_args['do_sample'], f'{pl.i("do_sample")} must be False for contrastive search'
            else:
                generate_args['do_sample'] = False
        else:  # seems to repeat itself
            assert strategy == 'beam'
            generate_args = dict(num_beams=3) | generate_args
            assert generate_args['num_beams'] > 1, f'{pl.i("num_beams")} must >1 for beam-search generation'

            accepted_keys = ['do_sample', 'num_beams', 'num_beam_groups']

            if 'num_beam_groups' in generate_args:
                accepted_keys.append('diversity_penalty')

                assert generate_args['num_beam_groups'] != 1  # sanity check

                if 'do_sample' in generate_args:
                    assert not generate_args['do_sample'], f'{pl.i("do_sample")} must be False for diverse beam search '
                else:
                    generate_args['do_sample'] = False
            else:
                generate_args = dict(do_sample=True) | generate_args
                if generate_args['do_sample']:
                    accepted_keys += accepted_keys_sample
            assert (k in accepted_keys for k in generate_args)

        args |= generate_args
        if args['do_sample']:
            args['renormalize_logits'] = True
        prompt_colored = self.tokenizer.colorize(prompt)
        d_log = dict(mode=mode, strategy=strategy, args=generate_args | prompt_args, prompt=prompt_colored)
        self.logger.info(f'Generating with {pl.i(d_log)}')
        t = datetime.datetime.now()
        self.model.eval()
        output = self.model.generate(**inputs, **args)  # for now, generate one at a time
        self.logger.info(f'Model generation finished in {pl.i(fmt_delta(datetime.datetime.now() - t))}')

        multiple_sequence = generate_args.get('num_return_sequences', 1) > 1
        if multiple_sequence:
            if truncate_last_bar:
                output = [self._truncate_last_bar(o) for o in output]
            decoded_ = self.tokenizer.batch_decode(output, skip_special_tokens=False)
        else:
            assert len(output) == 1  # sanity check
            output = output[0]

            if truncate_last_bar:
                output = self._truncate_last_bar(output)
            decoded_ = self.tokenizer.decode(output, skip_special_tokens=False)
        title_ = f'{save}\ngenerated' if save else 'generated'
        fnm_ = None
        save_path_ = None
        if save:
            str_args = MusicGenerator.args2fnm(dict(strategy=strategy) | args | prompt_args)
            save_path_ = u.eval_path
            if save_dir:
                save_path_ = os_join(save_path_, save_dir)
                os.makedirs(save_path_, exist_ok=True)
            fnm_ = title_.replace('\n', '_')
            fnm_ = f'{now(for_path=True)}_{fnm_}_{str_args}'

        def _single_post_gen(decoded: str = None, title: str = None, fnm: str = None, save_path: str = None):
            score = self.mc.str2score(
                decoded, omit_eos=True, title=title, pitch_kind=self.pitch_kind,
                check_duration_match='each-other'
            )
            if save:

                with open(os_join(save_path, f'{fnm}.json'), 'w') as f:
                    d_log['prompt'] = prompt  # remove color
                    json.dump(dict(meta=d_log, generation_args=args, generated=decoded), f, indent=4)
                try:
                    # TODO: `makeNotation` False always breaks on GL
                    score.write(
                        fmt='mxl', fp=os_join(save_path, f'{fnm}.mxl'),
                        # `makeNotations` disabled any clean-up by music21, intended to remove `tie`s added
                        makeNotation=False
                    )
                except Exception as e:
                    vocab = self.mc.vocabs.degree if self.augment_key else self.mc.vocabs.midi
                    err_str = f'Failed to render MXL from decoded output {vocab.colorize_tokens(decoded)}'
                    exc = traceback.format_exc()
                    raise ValueError(f'{err_str} due to exception: \n{pl.s(exc, c="r")}') from e
            else:
                score.show()
        if multiple_sequence:
            for i, dec in enumerate(decoded_, start=1):
                _single_post_gen(decoded=dec, title=f'{title_}_{i}', fnm=f'{fnm_}_{i}', save_path=save_path_)
        else:
            _single_post_gen(decoded=decoded_, title=title_, fnm=fnm_, save_path=save_path_)


def get_performance(model):
    """
    Get NTP IKR on the eval set
    """
    # TODO
    pass


if __name__ == '__main__':
    import musicnlp.util.music as music_util

    md_k = md_nm, ds_nm, ep_nm, desc = 'transf-xl', 'All', '128ep', 'small_long-seq'
    mic(md_nm, ds_nm, ep_nm, desc)

    # pch_kd = 'midi'
    pch_kd = 'degree'
    tk_args = dict(pitch_kind=pch_kd)

    # wp = True
    wp = False
    tk_args['fnm'] = '22-11-26_WordPiece-Tokenizer_{dnm=all}_{vsz=262144, n=178825, pch=d, aug-key=T}'

    md = 'full'
    mdl = load_trained(model_key=md_k, mode=md)
    sv_dir = f'{md_nm}_{ds_nm}_{ep_nm}_{desc}'
    # save_dir_ = 'reformer-base, 14/32ep'
    # mic(get_model_num_trainable_parameter(mdl))
    # step vocab for `MusicConverter::mxl2str`

    key_aug = True
    mg = MusicGenerator(
        model=mdl, mode=md, tokenizer_args=tk_args, augment_key=True, wordpiece_tokenize=wp,
        # pick_key='first-2'
        pick_key='sample'
    )

    def explore_generate_unconditional():
        from musicnlp.vocab import key_str2enum
        # as in `CTRL` paper
        # still repeats itself, and notion of penalizing repetition with music generation?
        # gen_args = dict(repetition_penalty=1.2)
        # how to set hyperparameters?; smaller k for smaller vocab size
        # gen_args = dict(temperature=1, top_k=16)
        # gen_args = dict(top_k=32, top_p=0.9)
        gen_args = dict(top_k=8, penalty_alpha=0.6)
        keys = list(key_str2enum.keys())
        prompt = dict(insert_key=True, key=random.choice(keys))
        mg(mode='unconditional', strategy='sample', generate_args=gen_args, prompt_args=prompt)
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
            # 'Canon piano', 'Piano Sonata', 'Für Elise', 'Symphony No.5', 'Flower Duet', 'The Marriage of Figaro',
            # 'Serenade No. 13', 'Serenade No. 13', 'K. 448', 'William Tell', 'Ave Maria', 'Hallelujah',

            # Re-run those with a different #bar in prompt, see `fnm2bar`
            'Merry Go Round of Life', 'Merry Go Round of Life',

            # 'Shape of You', '平凡之路', '平凡之路', 'Faded', 'Señorita', 'Sugar', 'Something Just Like This',
            # 'See You Again', 'Rolling in the Deep', 'Despacito',
            # '走马', '告白气球', '演员', '飘向北方', '年少有为', '丑八怪', '李白', '挪威的森林',

            # "Stayin' Alive", 'Careless Whisper', 'Take Me Home Country Roads', 'House of the Rising Sun'
        ]
        fnm2bar = {
            'Merry Go Round of Life': 4,
            '平凡之路': 4,
            'Flower Duet': 4,
            'Serenade No. 13': 4
        }

        strat = 'sample'
        # strat = 'contrastive'
        # strat = 'beam'
        if strat == 'sample':
            # gen_args = dict(top_k=16, top_p=0.75)  # this set up causes repetitions early on
            # gen_args = dict(top_k=32, top_p=0.95)
            # gen_args = dict(top_k=32, top_p=0.9)  # Kinda good for `All`
            # gen_args = dict(top_k=64, top_p=0.9)
            # gen_args = dict(top_k=32, top_p=0.75)  # Good w/ `P&M`, and 5-16 All
            # gen_args = dict(top_k=32, top_p=0.85)

            gen_args = dict(top_k=8)
            # gen_args = dict(top_k=16)
            # gen_args = dict(top_k=6, temperature=1.2)
            # gen_args = dict(top_k=32)
            # gen_args = dict(top_k=64, temperature=2.0)

            # gen_args = dict(top_p=0.75)
            # gen_args = dict(top_p=0.85)
            # gen_args = dict(top_p=0.95)  # Too much repetition if model trained well enough, e.g. NTP acc = 73
            # gen_args = dict(top_p=0.85, repetition_penalty=1.2)  # penalty as in CTRL paper
        elif strat == 'contrastive':  # TODO: doesn't seem to work w/ TransformerXL
            # gen_args = dict(top_k=32, penalty_alpha=0.3)
            gen_args = dict(top_k=16, penalty_alpha=0.3)
            # gen_args = dict(top_k=8, penalty_alpha=0.5)
            # gen_args = dict(top_k=8, penalty_alpha=0.6)
            # gen_args = dict(top_k=10, penalty_alpha=0.6)
            # gen_args = dict(top_k=16, penalty_alpha=0.6)
            # gen_args = dict(top_k=12, penalty_alpha=0.6)
        else:
            assert strat == 'beam'
            gen_args = dict(num_beams=3, do_sample=True, top_k=64)

            # gen_args = dict(num_beams=3, num_beam_groups=3, num_return_sequences=3, diversity_penalty=0.6)
        mic(strat, gen_args)

        # n_bar = 4
        n_bar = 8

        fnm_pattern = re.compile(r'^(?P<date>\d{2}-\d{2}-\d{2})_(?P<fnm>.*)_(?P<mode>{md=.})$')

        for fnm in fnms:
            path = music_util.get_my_example_songs(k=fnm, extracted=True, postfix='{md=f}')
            fnm = stem(path)
            n_bar = fnm2bar.pop(fnm, n_bar)
            m = fnm_pattern.match(fnm)
            assert m is not None
            fnm = m.group('fnm')
            prompt = dict(path=path, n_bar=n_bar, insert_key=key_aug, pitch_shift=pch_sft)
            mic(prompt)

            def call():
                mg(
                    mode='conditional', strategy=strat, generate_args=gen_args, prompt_args=prompt,
                    save=fnm, save_dir=sv_dir
                )
            if batched:
                try:
                    call()
                except Exception as e:
                    print(f'Failed to generate {pl.i(fnm)} due to exception: \n{e}')
            else:
                call()
    export_generated(batched=True)

    def eval_ikr():
        md_sz = 'debug'
    # eval_ikr()

    def fix_prior_fnm_not_contrastive():
        import re
        import shutil

        dnm = 'transf-xl_All_128ep_no-ch-mix'
        path = os_join(u.eval_path, dnm)
        fls = sorted(glob.iglob(os_join(path, '*.json')))
        mic(fls)

        pa_pattern = re.compile(r'(?P<before>.*)(, pa=0\.6)(?P<after>.*)')
        for fl in fls:
            if 'pa' in fl:
                m = pa_pattern.match(fl)
                assert m is not None
                # mic(m.group('before'), m.group('after'))

                new_fnm = m.group('before') + m.group('after')
                shutil.move(fl, new_fnm)
                # raise NotImplementedError
    # fix_prior_fnm_not_contrastive()
