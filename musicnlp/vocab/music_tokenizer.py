from typing import List, Tuple, Dict, Union, Optional, Iterable

from tokenizers import AddedToken
from transformers.tokenization_utils import PreTrainedTokenizer

from musicnlp.vocab.music_vocab import VocabType, MusicVocabulary


class MusicTokenizer(PreTrainedTokenizer):
    """
    Conversion between music tokens & int ids

    For integration with HuggingFace

    Note that there are **no special tokens**
    """
    model_input_names = ['input_ids']  # Per `TransfoXLTokenizer`

    def __init__(self, precision: int = 5, **kwargs):
        """
        :param precision: See musicnlp.preprocess.music_extractor
        """
        super().__init__(**kwargs)
        # Model max length undefined, for infinite input length; See `tokenization_utils_base`
        if self.model_max_length == int(1e30):
            self.model_max_length = 4096  # TODO: subject to change?

        self.precision = precision
        self.vocab = MusicVocabulary(precision=precision, color=False)
        self.spec_toks_enc, self.spec_toks_dec = dict(), dict()
        # if add_pad:
        #     self._add_pad_token()
        self.pad_token = self.vocab.pad  # override
        self.eos_token = self.vocab.end_of_song
        # self.add_special_tokens(dict(pad_token=self.vocab.pad, eos_token=self.vocab.end_of_song))
        # from stefutil import mic
        # mic(self.pad_token, self.eos_token)

        self.sob_token = self.vocab.start_of_bar
        self.uncommon_time_sig_token = self.vocab.uncommon_time_sig
        self.uncommon_low_tempo_token = self.vocab.uncommon_low_tempo
        self.uncommon_high_tempo_token = self.vocab.uncommon_high_tempo
        self.uncommon_duration_token = self.vocab.uncommon_duration
        self.sob_token_id = self._convert_token_to_id(self.sob_token)
        self.uncommon_time_sig_token_id = self._convert_token_to_id(self.uncommon_time_sig_token)
        self.uncommon_low_tempo_token_id = self._convert_token_to_id(self.uncommon_low_tempo_token)
        self.uncommon_high_tempo_token_id = self._convert_token_to_id(self.uncommon_high_tempo_token)
        self.uncommon_duration_token_id = self._convert_token_to_id(self.uncommon_duration_token)

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

    def ids2pitches(self, ids: Union[Iterable[int], Iterable[str]]) -> List[int]:
        """
        :param ids: token ids for a score, or split tokens
        :return: compact representation of all pitch (midi, see `MusicVocabulary`) in the score
        """
        return [self.vocab.compact(i) for i in ids if self.vocab.type(i) == VocabType.pitch]


if __name__ == '__main__':
    from stefutil import *

    from musicnlp.preprocess import get_dataset

    # mode = 'melody'
    mode = 'full'
    if mode == 'melody':
        fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-04'
        # fnm = 'musicnlp music extraction, dnm=LMD, n=176640, meta={mode=melody, prec=5, th=1}, 2022-05-27_15-23-20'
    else:
        # fnm = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=full, prec=5, th=1}, 2022-08-02_20-11-17'
        fnm = 'musicnlp music extraction, dnm=MAESTRO, n=1276, meta={mode=full, prec=5, th=1}, 2022-08-02_20-12-23'

    def implementation_check():
        dsets = get_dataset(fnm)
        # ic(dset, dset[:2])
        tkzer = MusicTokenizer(model_max_length=12)
        # tkzer = MusicTokenizer()
        mic(tkzer, tkzer.model_max_length, len(tkzer))
        txt = dsets[1]['score']
        # txt = dset[:3]['text']
        # Turning off both `padding` & `truncation`, and the token ids too long warning appears
        input_ = tkzer(txt, padding='max_length', truncation=True)
        # ic(input_)
        # ic(len(input_['input_ids']))
        ids_ = input_['input_ids']
        mic(input_, ids_, tkzer.decode(ids_))
    # implementation_check()

    def check_one_pass_all_data():
        """
        Make sure tokenization doesn't break
        """
        from tqdm.auto import trange
        dsets = get_dataset(fnm)
        tokenizer = MusicTokenizer(model_max_length=4096*16)
        for split, dset in dsets.items():
            it = trange(len(dset), desc=f'Tokenizing {split}', unit='song')
            for i in it:
                inputs = dset[i]
                ids = tokenizer(inputs['score'])['input_ids']
                it.set_postfix(len=len(ids))
    check_one_pass_all_data()


    def check_pad_n_eos():
        tkzer = MusicTokenizer(model_max_length=12)
        mic(tkzer.eos_token_id, tkzer.pad_token_id, tkzer.eos_token, tkzer.pad_token)
        vocab = tkzer.vocab
        mic(vocab.t2i(vocab.end_of_song))
    # check_pad_n_eos()

    def sanity_check_uncom():
        """
        A small ratio of tokens should be set to uncommon
        """
        import numpy as np

        from tqdm.auto import tqdm
    
        from musicnlp.util import sconfig

        tkzer = MusicTokenizer(model_max_length=int(1e20))
        np.random.seed(sconfig('random-seed'))

        n = 4096 * 4
        dsets = get_dataset(fnm)
        for split, dset in dsets.items():
            n_dset = len(dset)
            if n_dset > n:
                idxs = np.random.choice(n_dset, size=n, replace=False)
                for i in tqdm(idxs, desc=split):
                    txt = dset[int(i)]['score']
                    mic(tkzer(txt))
                    exit(1)
            else:
                for row in tqdm(dset, desc=split):
                    txt = row['score']
                    tkzer(txt)
    # sanity_check_uncom()

    def check_n_note_in_tup():
        from collections import Counter
        from tqdm.auto import tqdm

        from musicnlp.vocab.elm_type import ElmType
        from musicnlp.preprocess.dataset import load_songs
        from musicnlp.postprocess import MusicConverter
        mc = MusicConverter()

        pop = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-04'
        mst = 'musicnlp music extraction, dnm=MAESTRO, n=1276, meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-28'
        lmd = 'musicnlp music extraction, dnm=LMD, n=176640, meta={mode=melody, prec=5, th=1}, 2022-05-27_15-23-20'
        dnms = [pop, mst, lmd]

        def song2n_note(t: str) -> List[int]:
            elms = mc.str2notes(t)
            tups = [e.meta for e in elms if e.type == ElmType.tuplets]
            return [len(t[0]) for t in tups]

        def count(songs_):
            c_ = Counter()
            for s in tqdm(songs_):
                c_.update(song2n_note(s['score']))
            return c_

        each = True
        if each:
            for path in dnms:
                mic(path)
                mic(count(load_songs(path)))
        else:
            songs = sum([load_songs(d) for d in dnms], start=[])
            mic(len(songs))
            c = count(songs)
            mic(c)
    # check_n_note_in_tup()
