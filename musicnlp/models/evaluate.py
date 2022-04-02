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

    def __call__(self, mode: str, to_score: bool = False):
        modes = ['conditional', 'unconditional-greedy', 'unconditional-topk']
        assert mode in modes, f'Invalid mode: expect one of {logi(modes)}, got ({logi(mode)})'

        # ic(tokenizer, tokenizer.pad_token_id, tokenizer.eos_token_id)

        ts, tp = self.vocab.uncompact(VocabType.time_sig, (4, 4)), self.vocab.uncompact(VocabType.tempo, 120)
        # prompt = ' '.join([ts, tp])
        prompt = ' '.join([ts])
        inputs = self.tokenizer(prompt, return_tensors='pt')
        ic(inputs)  # as in `CTRL` paper
        if 'unconditional' in mode:
            if 'greedy' in mode:  # still repeats itself; and notion of penalizing repetition
                outputs = self.model.generate(**inputs, max_length=self.max_len, repetition_penalty=1.2)
            else:  # topk
                outputs = self.model.generate(
                    **inputs, max_length=self.max_len, do_sample=True,
                    # how to set hyperparameters?; smaller k for smaller vocab size
                    temperature=1, top_k=16,
                )
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
        mg(mode='unconditional-topk')
    trained_generate()

