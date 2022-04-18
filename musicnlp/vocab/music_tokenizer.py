from typing import List, Tuple, Dict, Union, Optional

from tokenizers import AddedToken
from transformers.tokenization_utils import PreTrainedTokenizer

from musicnlp.vocab.music_vocab import MusicVocabulary


class MusicTokenizer(PreTrainedTokenizer):
    """
    Conversion between music tokens & int ids

    For integration with HuggingFace

    Note that there are **no special tokens**
    """
    model_input_names = ['input_ids']  # Per `TransfoXLTokenizer`

    def __init__(self, precision: int = 5, deprecated: bool = False, **kwargs):
        super().__init__(**kwargs)
        # Model max length undefined, for infinite input length; See `tokenization_utils_base`
        if self.model_max_length == int(1e30):
            self.model_max_length = 4096  # TODO: subject to change?

        self.precision = precision
        self.vocab = MusicVocabulary(precision=precision, color=False, deprecated=deprecated)
        self.spec_toks_enc, self.spec_toks_dec = dict(), dict()
        # self._add_special_token(self.vocab.pad)
        self.pad_token, self.eos_token = self.vocab.pad, self.vocab.end_of_song
        self.sob_token = self.vocab.start_of_bar
        self.sob_token_id = self._convert_token_to_id(self.sob_token)

    def _add_special_token(self, tok):
        assert tok not in self.spec_toks_enc
        id_ = self.vocab_size
        self.spec_toks_enc[tok] = id_  # Assign the next coming value; vocab size automatically increments
        self.spec_toks_dec[id_] = tok

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        raise ValueError('In LMTTokenizer._add_tokens')

    def get_vocab(self) -> Dict[str, int]:
        raise ValueError('In LMTTokenizer.get_vocab')

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        raise ValueError('In LMTTokenizer.save_vocabulary')

    def _tokenize(self, text, **kwargs):
        return text.split()  # Each word in vocab is split by space; TODO: special token handling?

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.spec_toks_enc)

    def _convert_token_to_id(self, token):
        return self.spec_toks_enc[token] if token in self.spec_toks_enc else self.vocab.t2i(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.spec_toks_dec[index] if index in self.spec_toks_dec else self.vocab.i2t(index)


if __name__ == '__main__':
    from icecream import ic

    ic.lineWrapWidth = 400

    from musicnlp.preprocess import get_dataset

    fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-04-16_20-28-47'
    dset = get_dataset(fnm)

    def implementation_check():
        # ic(dset, dset[:2])
        tkzer = MusicTokenizer(model_max_length=12)
        # tkzer = MusicTokenizer()
        ic(tkzer, tkzer.model_max_length, len(tkzer))
        txt = dset[1]['score']
        # txt = dset[:3]['text']
        # Turning off both `padding` & `truncation`, and the token ids too long warning appears
        input_ = tkzer(txt, padding='max_length', truncation=True)
        # ic(input_)
        # ic(len(input_['input_ids']))
        ids_ = input_['input_ids']
        ic(input_, ids_, tkzer.decode(ids_))
    implementation_check()

    def check_pad_n_eos():
        tkzer = MusicTokenizer(model_max_length=12)
        ic(tkzer.eos_token_id, tkzer.pad_token_id, tkzer.eos_token, tkzer.pad_token)
        vocab = tkzer.vocab
        ic(vocab.t2i(vocab.end_of_song))
    # check_pad_n_eos()
