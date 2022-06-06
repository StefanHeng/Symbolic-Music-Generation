"""
Combine individual music token in `music_vocab`, treating each token as a character as in WordPiece tokenizer training
    i.e. Base vocab is tokens

Intended to trade sequence length with vocabulary size
    The vanilla tokenizer takes up
"""

import json
from os.path import join as os_join
from typing import List, Dict, Iterable, Union

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers import pre_tokenizers, models, decoders

from stefutil import *
from musicnlp.util import *
from musicnlp.vocab.music_vocab import MusicVocabulary
from musicnlp.vocab.music_tokenizer import MusicTokenizer


class Score2Chars:
    """
    To fit to existing WordPiece training, mapping between
        1) my music `score` format and 2) sequence of contiguous characters
    """
    def __init__(self, vocab: MusicVocabulary, chars: List[str] = None):
        """
        :param vocab: Handles music vocabulary processing, such as mapping from token to ordinal/id
        :param chars: A list of characters
            Intended for mapping each ordinal,
        """
        self.vocab = vocab
        if chars:
            assert len(chars) == len(vocab)
        else:
            chars = Score2Chars.get_uni_chars(len(vocab))
        self.dec_chars = chars
        self.enc_chars = {c: i for i, c in enumerate(chars)}

    @staticmethod
    def get_uni_chars(n: int) -> List[str]:
        """
        :return: A list of mostly-printing friendly unicode characters
        """
        strt, end = 0x0021, 0x02FF
        assert 0 < n <= end - strt + 1
        return [chr(i) for i in range(strt, strt + n)]

    def __call__(self, s: Union[str, List[str]], clean: bool = True) -> str:
        """
        score => chars
        """
        toks = s.split() if isinstance(s, str) else s
        if clean:
            toks = [self.vocab.clean_uncommon_token(t) for t in toks]
        return ''.join([self.dec_chars[self.vocab.tok2id[tok]] for tok in toks])

    def decode(self, s: str) -> str:
        """
        chars => score
        """
        return ' '.join([self.vocab.id2tok[self.enc_chars[c]] for c in s])

    def trained_tok2tok(self, tok: str, continuing_subword_prefix: str = '##') -> str:
        if tok.startswith(continuing_subword_prefix):
            tok = tok[len(continuing_subword_prefix):]
            return f'{continuing_subword_prefix}{self.decode(tok)}'
        else:
            return self.decode(tok)

    def char_vocab2vocab(self, vocab: Dict[str, int], continuing_subword_prefix: str = '##'):
        """
        The HF trained tokenizer in char, to the human-readable, my music token representation
        """
        return {self.trained_tok2tok(tok, continuing_subword_prefix): i for tok, i in vocab.items()}


class WordPieceMusicTrainer:
    """
    Wrapper for training music-score representation with WordPiece tokenizer
    """
    def __init__(self, vocab: MusicVocabulary):
        self.vocab = vocab
        self.s2c = Score2Chars(mv)

    def __call__(self, vocab_size: int = 2**13, songs: Iterable[str] = None, save: Union[bool, str] = None):
        # TODO: don't need to pass `vocab` to `WordPiece`?
        # every token should be known
        # TODO: What is `max_input_chars_per_word`? set no lim
        tokenizer = Tokenizer(model=models.WordPiece(vocab=None, max_input_chars_per_word=int(1e10)))
        tokenizer.decoder = decoders.WordPiece()
        trainer = WordPieceTrainer(vocab_size=vocab_size, initial_alphabet=self.s2c.dec_chars, show_progress=True)
        tokenizer.train_from_iterator((self.s2c(s) for s in songs), trainer=trainer)
        if save:
            if isinstance(save, bool):
                save = 'Word-Piece-Music-Tokenizer'
            now_ = now(for_path=True)
            tokenizer.save(os_join(u.tokenizer_path, f'{now_}_{save}.json'))
            fnm_meta = f'{save}_music_meta'
            with open(os_join(u.tokenizer_path, f'{now_}_{fnm_meta}.json'), 'w') as f:
                json.dump(dict(  # For reconstructing class properties, see `WordPieceMusicTokenizer`
                    music_vocab=dict(prec=self.vocab.precision),
                    score2chars=dict(chars=self.s2c.dec_chars),
                ), f, indent=4)
        return tokenizer

    def music_vocab(self, tokenizer: Tokenizer) -> Dict[str, int]:
        return self.s2c.char_vocab2vocab(tokenizer.get_vocab())


class MyPreTrainedTokenizerFast(PreTrainedTokenizerFast):
    """
    Override tokenization return vars
    """
    model_input_names = ['input_ids']


class WordPieceMusicTokenizer(MusicTokenizer):
    def __init__(self, tokenizer: Tokenizer, precision: int = 5, chars: List[str] = None, **kwargs):
        """
        :param tokenizer: A trained WordPiece tokenizer on characters
        """
        super().__init__(precision=precision, **kwargs)
        _tokenizer = tokenizer
        self._tokenizer = MyPreTrainedTokenizerFast(
            tokenizer_object=_tokenizer, pad_token=self.pad_token, eos_token=self.eos_token
        )
        self.s2c = Score2Chars(self.vocab, chars)

    @classmethod
    def from_file(cls, fnm: str, output_path: str = u.tokenizer_path):
        _tokenizer = Tokenizer.from_file(os_join(output_path, f'{fnm}.json'))
        with open(os_join(output_path, f'{fnm}_music_meta.json'), 'r') as f:
            meta = json.load(f)
        prec, chars = meta['music_vocab']['prec'], meta['score2chars']['chars']
        return cls(_tokenizer, precision=prec, chars=chars)

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return self._tokenizer(self.s2c(text, clean=True), **kwargs)
        else:
            raise NotImplementedError('Not implemented for iterable input')

    def decode(self, token_ids, **kwargs):
        decoded = self._tokenizer.decode(token_ids, **kwargs, clean_up_tokenization_spaces=False)
        if isinstance(decoded, str):
            return self.s2c.decode(decoded)
        else:
            raise NotImplementedError('Not implemented for iterable input')


if __name__ == '__main__':
    from icecream import ic

    from musicnlp.preprocess.dataset import load_songs

    sample_txt = 'TimeSig_1/4 Tempo_120 <bar> p_7/4 d_1/4 p_r d_1/4 p_12/4 d_1/4 p_r d_1/4 <bar> p_2/5 d_1/4 p_r ' \
                 'd_1/4 p_4/5 d_1/2 <bar> p_4/5 d_1 <bar> p_4/5 d_1/2 p_5/5 d_1/4 p_r d_1/4 <bar> p_4/5 d_1 <bar> ' \
                 'p_4/5 d_1/2 p_2/5 d_1/2 <bar> p_2/5 d_1 <bar> p_7/5 d_1 <bar> p_7/5 d_1/2 p_r d_1/2 <bar> p_12/4 ' \
                 'd_1 <bar> p_12/4 d_1 <bar> p_2/5 d_1/4 p_r d_1/4 p_12/4 d_1/4 p_r d_1/4 <bar> p_12/4 d_1 <bar> ' \
                 'p_11/4 d_1 <bar> p_11/4 d_1 <bar> p_11/4 d_1 <bar> p_11/4 d_1/2 p_4/4 d_1/2 <bar> p_5/5 d_1 <bar> ' \
                 'p_5/5 d_1 <bar> p_5/5 d_1 <bar> p_5/5 d_1 <bar> p_2/5 d_1 <bar> p_2/5 d_1 <bar> p_4/5 d_1 <bar> ' \
                 'p_4/5 d_1/2 p_r d_1/2 <bar> p_5/5 d_1 <bar> p_5/5 d_1 <bar> p_5/5 d_1 <bar> p_4/5 d_1/2 p_2/5 d_1/2 '\
                 '<bar> p_2/5 d_1 <bar> p_2/5 d_1 <bar> p_11/4 d_1 <bar> p_12/4 d_1/2 p_2/5 d_1/4 p_r d_1/4 <bar> ' \
                 'p_4/5 d_1 <bar> p_12/4 d_1 <bar> p_5/5 d_1/4 p_r d_1/4 p_4/5 d_1/2 <bar> p_4/5 d_1/2 p_12/4 d_1/4 ' \
                 'p_r d_1/4 <bar> p_11/4 d_1/2 p_11/4 d_1/2 <bar> p_11/4 d_1/2 p_7/5 d_1/2 <bar> p_7/5 d_1 <bar> ' \
                 'p_2/5 d_1/2 p_7/3 d_1/4 p_r d_1/4 <bar> p_12/4 d_1 <bar> p_12/4 d_1 <bar> p_4/5 d_1/2 p_4/5 d_1/2 ' \
                 '<bar> p_4/5 d_1/2 p_9/4 d_1/2 <bar> p_11/4 d_1/2 p_9/5 d_1/2 <bar> p_9/5 d_1/2 p_7/5 d_1/2 <bar> ' \
                 'p_7/5 d_1 <bar> p_7/5 d_1/2 p_12/4 d_1/4 p_r d_1/4 '

    def sanity_check_split():
        pre_tokenizer = pre_tokenizers.WhitespaceSplit()  # split on whitespace only
        ic(pre_tokenizer.pre_tokenize_str(sample_txt))
    # sanity_check_split()

    mv = MusicVocabulary()
    # vocab = list(mv.tok2id.keys())
    # ic(vocab, len(vocab))

    pop = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-04'
    # songs = [songs[0][:256], songs[1][:256]]
    # ic(type(songs))
    # ic(len(songs))
    # ic(type(songs[0]), len(songs[0]))

    def check_tokenize_train():
        unk = '[UNK]'
        tokenizer = Tokenizer(model=WordPiece(vocab=None, unk_token=unk, max_input_chars_per_word=int(1e10)))
        # input scores already cleaned, no normalizer needed
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = WordPieceTrainer(
            vocab_size=int(2e5), initial_alphabet=list(mv.tok2id.keys()), special_tokens=[unk], show_progress=True
        )
        songs = load_songs(pop)
        tokenizer.train_from_iterator(songs, trainer=trainer)
        ic(tokenizer.get_vocab_size())
        ic(tokenizer.get_vocab())
    # check_tokenize_train()

    def try_char_map():
        # chs = get_uni_chars(40)
        # ic(chs, len(chs))
        # exit(1)
        s2c = Score2Chars(mv)
        sample_txt_ = mv.clean_uncommon(sample_txt, return_joined=False)
        encoded = s2c(sample_txt_)
        ic(encoded)
        decoded = s2c.decode(encoded)
        ic(decoded)
        assert ' '.join(sample_txt_) == decoded
    # try_char_map()

    def train():
        from collections import Counter

        from tqdm.auto import tqdm

        vocab_size = 4096
        wmt = WordPieceMusicTrainer(mv)
        songs = load_songs(pop)
        # sv = False
        sv = True
        tokenizer = wmt(vocab_size=vocab_size, songs=songs, save=sv)

        s2c = wmt.s2c

        check_decode = False
        if check_decode:
            sample_txt_ = mv.clean_uncommon(sample_txt, return_joined=False)
            encoded = s2c(sample_txt_)
            encoded = tokenizer.encode(encoded).ids
            ic(tokenizer.decoder)
            ic(encoded)
            ic(tokenizer.decode(encoded))

        check_dist = False
        if check_dist:

            c = Counter()
            # for song in tqdm(songs):
            for song in tqdm(songs):
                song = mv.clean_uncommon(song)
                chars = tokenizer.encode(s2c(song))
                # ic(chars)
                # ic(type(chars))
                # ic(chars.ids)
                # ic(chars.tokens)
                c.update([s2c.trained_tok2tok(t) for t in chars.tokens])
                # c.update(chars.ids)
                # exit(1)
            # c = Counter({s2c.trained_tok2tok(tokenizer.id_to_token(i)): c for i, c in c.items()})
            ic(c)
    # train()

    def check_trained():
        fnm = '2022-06-06_16-36-11_Word-Piece-Music-Tokenizer'
        tokenizer = WordPieceMusicTokenizer.from_file(fnm)
        # ic(tokenizer)
        inputs = tokenizer(sample_txt)
        ic(inputs)
        ic(tokenizer.decode(inputs['input_ids']))
    check_trained()
