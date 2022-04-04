"""
Generate from trained reformer, no seed per `hash_seed`
"""

from transformers import ReformerModelWithLMHead

from musicnlp.util import *
from musicnlp.util.music_vocab import VocabType
from musicnlp.models.music_tokenizer import MusicTokenizer
from musicnlp.postprocess import MusicConverter


def load_trained(model_name: str, directory_name: str):
    path = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, model_name, directory_name, 'trained')
    return ReformerModelWithLMHead.from_pretrained(path)


class MusicGenerator:
    """
    Wraps a model for music generation
    """
    def __init__(self, model: ReformerModelWithLMHead):
        self.model = model
        self.max_len = model.config.max_position_embeddings
        self.tokenizer = MusicTokenizer(model_max_length=self.max_len)
        self.vocab = self.tokenizer.vocab
        self.converter = MusicConverter()

        self.eval_path = os.path.join(PATH_BASE, DIR_PROJ, 'evaluations')
        os.makedirs(self.eval_path, exist_ok=True)

    def __call__(
            self, mode: str, strategy: str, to_score: bool = False,
            generate_args: dict = None, prompt_args: dict = None,
            truncate_to_sob: bool = True, save: Union[bool, str] = False
    ):
        """
        :param mode: One of [`conditional`, `unconditional`]
            If `conditional`, expect either `prompt` as string or path to a `musicnlp` extracted MXL file
                in `condition_args`
        :param strategy: One of [`greedy`, `sample`]
            If `sample`, expect `topk`, `temperature`, `top_p` in `generate_args`
        :param truncate_to_sob: If True, truncate the generated tokens such that the generated ends in terms of bar
            Intended for converting to MXL
        :param save: If false, the score is shown
            Otherwise, should be a filename string, the generated score is saved to an MXL file
        """
        modes = ['conditional', 'unconditional-greedy', 'unconditional-sample']
        assert mode in modes, f'Invalid mode: expect one of {logi(modes)}, got ({logi(mode)})'
        strategies = ['greedy', 'sample']
        assert strategy in strategies, f'Invalid strategy: expect one of {logi(strategies)}, got ({logi(strategy)})'

        if generate_args is None:
            generate_args = dict()
        if prompt_args is None:
            prompt_args = dict()
        if mode == 'unconditional':
            ts, tp = self.vocab.uncompact(VocabType.time_sig, (4, 4)), self.vocab.uncompact(VocabType.tempo, 120)
            # prompt = ' '.join([ts, tp])
            # prompt = ' '.join([ts])
            prompt = ' '.join([ts, tp, self.vocab.start_of_bar])
        else:  # 'conditional'
            prompt, path = prompt_args.get('prompt', None), prompt_args.get('path', None)
            assert prompt is not None or path is not None, f'Expect either {logi("prompt")} or {logi("path")}'
            if prompt is None:
                n_bar = prompt_args.get('n_bar', 4)
                prompt = self.converter.mxl2str(path, n_bar=n_bar)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        args = dict(max_length=self.max_len)
        if strategy == 'greedy':
            if 'do_sample' in generate_args:
                assert not generate_args['do_sample'], f'{logi("do_sample")} must be False for greedy generation'
            else:
                generate_args['do_sample'] = False
        else:  # 'sample'
            if 'do_sample' in generate_args:
                assert generate_args['do_sample'], f'{logi("do_sample")} must be True for sample generation'
            else:
                generate_args['do_sample'] = True
        args |= generate_args
        outputs = self.model.generate(**inputs, **args)[0]  # for now, generate one at a time
        if truncate_to_sob:
            idxs_eob = torch.nonzero(outputs == self.tokenizer.sob_token_id).flatten().tolist()
            assert len(idxs_eob) > 0, f'No start of bar token found when {logi("truncate_to_sob")} enabled'
            outputs = outputs[:idxs_eob[-1]]  # truncate also that `sob_token`
        decoded = self.tokenizer.decode(outputs, skip_special_tokens=False)
        title = f'{save}-generated' if save is not None else None
        score = self.converter.str2score(decoded, omit_eos=True, title=title)  # incase model can't finish generation
        if save:
            # `makeNotations` disabled any clean-up by music21, intended to remove `tie`s added
            path = os.path.join(self.eval_path, f'{title}, {now(for_path=True)}.mxl')
            score.write(fmt='mxl', fp=path, makeNotation=False)
        else:
            score.show()


if __name__ == '__main__':
    from icecream import ic

    mdl = load_trained(model_name='reformer', directory_name='2022-04-01_09-40-48')
    ic(get_model_num_trainable_parameter(mdl))
    mg = MusicGenerator(mdl)

    def explore_generate_unconditional():
        # as in `CTRL` paper
        # still repeats itself, and notion of penalizing repetition with music generation?
        # gen_args = dict(repetition_penalty=1.2)
        # how to set hyperparameters?; smaller k for smaller vocab size
        # gen_args = dict(temperature=1, top_k=16)
        gen_args = dict(top_k=0, top_p=0.9)
        mg(mode='unconditional', strategy='sample', generate_args=gen_args)
    explore_generate_unconditional()

    def explore_generate_conditional():
        fnm = 'Merry Go Round'
        path = get_my_example_songs(k=fnm, extracted=True)
        gen_args = dict(topk=16, top_p=0.75)
        prompt = dict(path=path)
        mg(mode='conditional', strategy='sample', generate_args=gen_args, prompt_args=prompt, save=fnm)
    # explore_generate_conditional()

    def check_why_tie_in_output():
        incorrect_tie_fl = '/Users/stefanh/Desktop/incorrect tie.mxl'
        score = m21.converter.parse(incorrect_tie_fl)
        for bar in list(score.parts)[0][Measure]:
            for e in bar:
                if isinstance(e, (Note, Rest)):
                    ic(e, e.tie)
    # check_why_tie_in_output()

    def export_generated():
        fnms = ['Merry Go Round of Life', 'Shape of You']
        gen_args = dict(topk=16, top_p=0.75)
        for fnm in fnms:
            path = get_my_example_songs(k=fnm, extracted=True)
            prompt = dict(path=path)
            mg(mode='conditional', strategy='sample', generate_args=gen_args, prompt_args=prompt, save=fnm)
    # export_generated()
