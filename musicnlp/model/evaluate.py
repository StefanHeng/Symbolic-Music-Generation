"""
Generate from trained reformer, no seed per `hash_seed`
"""


from transformers import ReformerModelWithLMHead

from musicnlp.util import *
from musicnlp.util.music_vocab import VocabType
from musicnlp.model.music_tokenizer import MusicTokenizer


def load_trained(model_name: str, directory_name: str):
    path = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, model_name, directory_name, 'trained')
    # os.listdir(path)
    return ReformerModelWithLMHead.from_pretrained(path)


def generate_music(mdl: ReformerModelWithLMHead, mode: str):
    modes = ['conditional', 'unconditional']
    assert mode in modes, f'Invalid mode: expect one of {logi(modes)}, got ({logi(mode)})'

    max_len = mdl.config.max_position_embeddings
    tokenizer = MusicTokenizer(model_max_length=max_len)
    ic(tokenizer, tokenizer.pad_token_id, tokenizer.eos_token_id)
    vocab = tokenizer.vocab
    ts, tp = vocab.uncompact(VocabType.time_sig, (4, 4)), vocab.uncompact(VocabType.tempo, 120)
    prompt = ' '.join([ts, tp])
    inputs = tokenizer(prompt, return_tensors='pt')
    ic(inputs)
    outputs = mdl.generate(**inputs)
    decoded = tokenizer.decode(
        outputs[0], skip_special_tokens=False,
        # repetition_penalty=2.0,
        max_length=max_len
    )
    ic(outputs, decoded)


if __name__ == '__main__':
    from icecream import ic

    def trained_generate():
        mdl = load_trained(model_name='reformer', directory_name='2022-04-01_09-40-48')
        generate_music(mdl, mode='unconditional')
    trained_generate()
