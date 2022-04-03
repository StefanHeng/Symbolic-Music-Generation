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

    def __call__(self, mode: str, to_score: bool = False, generate_args: dict = None, condition_args: dict = None):
        """
        :param mode: If conditional, expect `topk`, `temperature`, `top_p` in `generate_args`
            If unconditional, expect either `prompt` as string or path to a `musicnlp` extracted MXL file
        """
        modes = ['conditional', 'unconditional-greedy', 'unconditional-sample']
        assert mode in modes, f'Invalid mode: expect one of {logi(modes)}, got ({logi(mode)})'

        if generate_args is None:
            generate_args = dict()
        if condition_args is None:
            condition_args = dict()
        if 'unconditional' in mode:
            ts, tp = self.vocab.uncompact(VocabType.time_sig, (4, 4)), self.vocab.uncompact(VocabType.tempo, 120)
            # prompt = ' '.join([ts, tp])
            # prompt = ' '.join([ts])
            prompt = ' '.join([ts, tp, self.vocab.start_of_bar])
            inputs = self.tokenizer(prompt, return_tensors='pt')
            args = dict(max_length=self.max_len)
            ic(inputs)
            if 'greedy' in mode:
                if 'do_sample' in generate_args:
                    assert not generate_args['do_sample'], f'{logi("do_sample")} must be False for greedy generation'
            elif 'sample' in mode:
                if 'do_sample' in generate_args:
                    assert generate_args['do_sample'], f'{logi("do_sample")} must be True for sample generation'
            args |= generate_args
            outputs = self.model.generate(**inputs, **args)
        else:
            assert 'conditional' in mode
            prompt, path = condition_args.get('prompt', None), condition_args.get('path', None)
            assert prompt is not None or path is not None, f'Expect either {logi("prompt")} or {logi("path")}'
            if prompt is None:
                n_bar = condition_args.get('n_bar', 4)
                prompt = self.converter.mxl2str(path, n_bar=n_bar)
            inputs = self.tokenizer(prompt, return_tensors='pt')
            # greedy decoding
            outputs = self.model.generate(
                **inputs, max_length=self.max_len
            )
        ic(prompt)
        ic(outputs.shape)
        decoded = self.tokenizer.decode(
            outputs[0], skip_special_tokens=False,
        )
        ic(outputs, decoded)
        scr = self.converter.str2score(decoded)
        # scr.show()


if __name__ == '__main__':
    from icecream import ic

    def trained_generate():
        mdl = load_trained(model_name='reformer', directory_name='2022-04-01_09-40-48')
        mg = MusicGenerator(mdl)

        # as in `CTRL` paper
        # still repeats itself, and notion of penalizing repetition with music generation?
        # gen_args = dict(repetition_penalty=1.2)
        # how to set hyperparameters?; smaller k for smaller vocab size
        # gen_args = dict(temperature=1, top_k=16)
        # gen_args = dict(top_k=0, top_p=0.9)
        # mg(mode='unconditional-sample', generate_args=gen_args)

        fnm = 'Merry Go Round'
        path = get_my_example_songs(k=fnm, extracted=True)
        gen_args = dict(path=path)
        mg(mode='conditional', generate_args=gen_args)
    trained_generate()
