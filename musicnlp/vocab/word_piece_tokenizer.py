"""
Combine individual music token in `music_vocab`, treating each token as a character as in WordPiece tokenizer training
    i.e. Base vocab is tokens

Intended to trade sequence length with vocabulary size
    The vanilla tokenizer takes up
"""

import json
from os.path import join as os_join
from typing import List, Dict, Union

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
    def __init__(self, vocab: MusicVocabulary, chars: List[str] = None, continuing_prefix: str = '##'):
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
        self.continuing_prefix = continuing_prefix

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

    def trained_tok2tok(self, tok: str) -> str:
        if tok.startswith(self.continuing_prefix):
            tok = tok[len(self.continuing_prefix):]
            return f'{self.continuing_prefix}{self.decode(tok)}'
        else:
            return self.decode(tok)

    def char_vocab2vocab(self, vocab: Dict[str, int]):
        """
        The HF trained tokenizer in char, to the human-readable, my music token representation
        """
        return {self.trained_tok2tok(tok): i for tok, i in vocab.items()}


class WordPieceMusicTrainer:
    """
    Wrapper for training music-score representation with WordPiece tokenizer
    """
    continuing_prefix = '##'

    def __init__(self, vocab: MusicVocabulary):
        self.vocab = vocab
        self.s2c = Score2Chars(vocab=mv, continuing_prefix=WordPieceMusicTrainer.continuing_prefix)

    def __call__(self, vocab_size: int = 2**13, songs: List[str] = None, save: Union[bool, str] = None):
        # TODO: don't need to pass `vocab` to `WordPiece`?
        # every token should be known
        # TODO: What is `max_input_chars_per_word`? set no lim
        logger = get_logger(self.__class__.__qualname__)
        d_log = {'vocab-size': vocab_size, '#song': len(songs)}
        logger.info(f'Training launched with {log_dict(d_log)}')

        tokenizer = Tokenizer(model=models.WordPiece(vocab=None, max_input_chars_per_word=int(1e10)))
        tokenizer.decoder = decoders.WordPiece(prefix=WordPieceMusicTrainer.continuing_prefix)
        trainer = WordPieceTrainer(
            vocab_size=vocab_size, initial_alphabet=self.s2c.dec_chars, show_progress=True,
            continuing_subword_prefix=WordPieceMusicTrainer.continuing_prefix
        )
        tokenizer.train_from_iterator((self.s2c(s) for s in songs), trainer=trainer)
        if save:
            if isinstance(save, bool):
                save = 'Word-Piece-Music-Tokenizer'
            now_ = now(for_path=True)
            fnm = f'{now_}_{save}, vsz={vocab_size}, n={len(songs)}'
            path_tok = os_join(u.tokenizer_path, f'{fnm}.json')
            tokenizer.save(path_tok)
            logger.info(f'{logi("Tokenizer")} saved to {logi(path_tok)}')
            path_meta = os_join(u.tokenizer_path, f'{fnm}_music_meta.json')
            with open(path_meta, 'w') as f:
                json.dump(dict(  # For reconstructing class properties, see `WordPieceMusicTokenizer`
                    music_vocab=dict(prec=self.vocab.precision),
                    score2chars=dict(chars=self.s2c.dec_chars),
                ), f, indent=4)
            logger.info(f'{logi("Tokenizer")} music metadata saved to {logi(path_meta)}')
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
        self.continuing_prefix = _tokenizer.decoder.prefix
        self.s2c = Score2Chars(vocab=self.vocab, chars=chars, continuing_prefix=self.continuing_prefix)

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

    # def _convert_token_to_id(self, token):
    #     raise NotImplementedError()

    def _convert_id_to_token(self, index: int) -> str:
        return self.s2c.decode(self._tokenizer.decode(index).removeprefix(self.continuing_prefix))


class _CheckTrainedMap:
    """
    **debugging**, see `check_trained`
    """
    def __init__(self, vocab: MusicVocabulary, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __call__(self, song_: str) -> List[int]:
        toks_ = self.vocab.clean_uncommon(song_, return_joined=False)
        song_ = ' '.join(toks_)
        return self.tokenizer(song_)['input_ids']


if __name__ == '__main__':
    from stefutil.prettier import mic

    mic.output_width = 128

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
        mic(pre_tokenizer.pre_tokenize_str(sample_txt))
    # sanity_check_split()

    mv = MusicVocabulary()
    # vocab = list(mv.tok2id.keys())
    # ic(vocab, len(vocab))

    pop = 'musicnlp music extraction, dnm=POP909, n=909, meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-04'
    mst = 'musicnlp music extraction, dnm=MAESTRO, n=1276, meta={mode=melody, prec=5, th=1}, 2022-05-20_14-52-28'
    lmd = 'musicnlp music extraction, dnm=LMD, n=176640, meta={mode=melody, prec=5, th=1}, 2022-05-27_15-23-20'
    # songs = [songs[0][:256], songs[1][:256]]
    # ic(type(songs))
    # ic(len(songs))
    # ic(type(songs[0]), len(songs[0]))

    def check_tokenize_train():
        unk = '[UNK]'
        tokenizer = Tokenizer(model=models.WordPiece(vocab=None, unk_token=unk, max_input_chars_per_word=int(1e10)))
        # input scores already cleaned, no normalizer needed
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = WordPieceTrainer(
            vocab_size=int(2e5), initial_alphabet=list(mv.tok2id.keys()), special_tokens=[unk], show_progress=True
        )
        songs = load_songs(pop)
        tokenizer.train_from_iterator(songs, trainer=trainer)
        mic(tokenizer.get_vocab_size())
        mic(tokenizer.get_vocab())
    # check_tokenize_train()

    def try_char_map():
        # chs = get_uni_chars(40)
        # ic(chs, len(chs))
        # exit(1)
        s2c = Score2Chars(mv)
        sample_txt_ = mv.clean_uncommon(sample_txt, return_joined=False)
        encoded = s2c(sample_txt_)
        mic(encoded)
        decoded = s2c.decode(encoded)
        mic(decoded)
        assert ' '.join(sample_txt_) == decoded
    # try_char_map()

    def train():
        from collections import Counter

        from tqdm.auto import tqdm

        dnms = [pop, mst, lmd]
        vocab_size, sv = None, None
        if len(dnms) == 1:
            vocab_size = 4096
            sv = 'Word-Piece-Music-Tokenizer, dnm=POP909'
        elif len(dnms) == 2:
            vocab_size = 4096 * 2
            sv = 'Word-Piece-Music-Tokenizer, dnm=POP & MST'
        elif len(dnms) == 3:
            vocab_size = 4096 * 4
            sv = 'Word-Piece-Music-Tokenizer, dnm=all'
        wmt = WordPieceMusicTrainer(mv)
        songs = load_songs(*dnms)
        # sv = False
        # sv = True
        tokenizer = wmt(vocab_size=vocab_size, songs=songs, save=sv)

        s2c = wmt.s2c

        check_decode = False
        if check_decode:
            sample_txt_ = mv.clean_uncommon(sample_txt, return_joined=False)
            encoded = s2c(sample_txt_)
            encoded = tokenizer.encode(encoded).ids
            mic(tokenizer.decoder)
            mic(encoded)
            mic(tokenizer.decode(encoded))

        check_dist = False
        # check_dist = True
        if check_dist:
            c = Counter()
            it = tqdm(songs)
            for song in it:
                toks = mv.clean_uncommon(song, return_joined=False)
                it.set_postfix(n_tok=len(toks))
                chars = tokenizer.encode(' '.join(toks))
                c.update([s2c.trained_tok2tok(t) for t in chars.tokens])
            mic(c)
    # train()

    def check_trained():
        from collections import Counter

        from tqdm.auto import tqdm

        fnm = '2022-06-06_17-39-34_Word-Piece-Music-Tokenizer, dnm=POP & MST, vsz=8192, n=2185'
        tokenizer = WordPieceMusicTokenizer.from_file(fnm)
        # inputs = tokenizer(sample_txt)
        # mic(tokenizer.decode(inputs['input_ids']))

        check_preserve = True  # sanity check, encoding & decoding, every token is still preserved
        check_dist = True

        # dnms = [pop]
        dnms = [lmd]
        # dnms = [pop, mst, lmd]
        songs = load_songs(*dnms)
        # concurrent = True
        concurrent = False
        if concurrent:
            map_single = _CheckTrainedMap(mv, tokenizer)
            lst_ids = conc_map(map_single, songs, with_tqdm=dict(chunksize=16), mode='process')
            c = Counter()
            for ids in tqdm(lst_ids):
                c.update(tokenizer.convert_ids_to_tokens(ids))
            mic(c)
        else:
            c = Counter()
            it = tqdm(songs)  # TODO: tokenizing long texts in MAESTRO take a long time...
            for song in it:
                toks = mv.clean_uncommon(song, return_joined=False)
                song = ' '.join(toks)
                it.set_postfix(n_tok=len(toks))
                ids = tokenizer(song)['input_ids']
                if check_preserve:
                    decoded = tokenizer.decode(ids)
                    assert song == decoded
                if check_dist:
                    c.update(tokenizer.convert_ids_to_tokens(ids))
            mic(c)
    check_trained()
