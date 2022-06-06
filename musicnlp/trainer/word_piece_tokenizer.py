"""
Combine individual music token in `music_vocab`, treating each token as a character as in WordPiece tokenizer training
    i.e. Base vocab is tokens

Intended to trade sequence length with vocabulary size
    The vanilla tokenizer takes up
"""

from typing import List, Union

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers import pre_tokenizers


if __name__ == '__main__':
    from icecream import ic

    from musicnlp.vocab import MusicVocabulary
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
    songs = load_songs(pop)
    # songs = [songs[0][:256], songs[1][:256]]
    # ic(type(songs))
    # ic(len(songs))
    # ic(type(songs[0]), len(songs[0]))

    def check_tokenize_train():
        # TODO: There shouldn't be an `unknown` token?
        unk = '[UNK]'
        # TODO: What is `max_input_chars_per_word`? set no lim
        tokenizer = Tokenizer(model=WordPiece(vocab=None, unk_token=unk, max_input_chars_per_word=int(1e10)))
        # input scores already cleaned, no normalizer needed
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = WordPieceTrainer(
            vocab_size=int(2e5), initial_alphabet=list(mv.tok2id.keys()), special_tokens=[unk], show_progress=True
        )
        tokenizer.train_from_iterator(songs, trainer=trainer)
        ic(tokenizer.get_vocab_size())
        ic(tokenizer.get_vocab())
    # check_tokenize_train()

    def try_char_map():
        def get_uni_chars(n: int):
            strt, end = 0x0021, 0x02FF
            assert 0 < n <= end-strt + 1  # A list of mostly-printing friendly chars
            return [chr(i) for i in range(strt, strt+n)]

        # chs = get_uni_chars(40)
        # ic(chs, len(chs))
        # exit(1)

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
                    chars = get_uni_chars(len(vocab))
                self.dec_chars = chars
                self.enc_chars = {c: i for i, c in enumerate(chars)}

            def __call__(self, s: Union[str, List[str]]) -> str:
                """
                score => chars
                """
                toks = s.split() if isinstance(s, str) else s
                return ''.join([self.dec_chars[self.vocab.tok2id[tok]] for tok in toks])

            def decode(self, s: str) -> str:
                """
                chars => score
                """
                return ' '.join([self.vocab.id2tok[self.enc_chars[c]] for c in s])

        s2c = Score2Chars(mv)
        sample_txt_ = mv.clean_uncommon(sample_txt, return_joined=False)
        encoded = s2c(sample_txt_)
        ic(encoded)
        decoded = s2c.decode(encoded)
        ic(decoded)
        assert ' '.join(sample_txt_) == decoded
        exit(1)

        # every token should be known
        tokenizer = Tokenizer(model=WordPiece(vocab=None, max_input_chars_per_word=int(1e10)))
        # tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=r' ', behavior='isolated', invert=False)
        # ic(tokenizer.pre_tokenizer.pre_tokenize_str(sample_txt))
        exit(1)
        trainer = WordPieceTrainer(vocab_size=int(2e5), initial_alphabet=vocab_, show_progress=True)
        tokenizer.train_from_iterator(songs, trainer=trainer)
        ic(tokenizer.get_vocab_size())
        ic(tokenizer.get_vocab())
    try_char_map()
