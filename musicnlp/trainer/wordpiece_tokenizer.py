"""
Combine individual music token in `music_vocab`, treating each token as a character as in WordPiece tokenizer training
    i.e. Base vocab is tokens

Intended to trade sequence length with vocabulary size
    The vanilla tokenizer takes up
"""

import json
from os.path import join as os_join
from typing import List, Dict, Union, Iterable, Any

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers import pre_tokenizers, models, decoders

from stefutil import *
from musicnlp.util import *
from musicnlp.vocab.music_vocab import MusicVocabulary, VocabType, WORDPIECE_CONTINUING_PREFIX
from musicnlp.vocab.music_tokenizer import MusicTokenizer
from musicnlp.preprocess import KeyInsert, PitchShift


def get_uni_chars_cache() -> List[str]:
    """
    :return: A list of mostly-printing friendly unicode characters
    """
    if not hasattr(get_uni_chars_cache, 'ranges'):
        # set of characters got from https://en.wikipedia.org/wiki/Latin_script_in_Unicode,
        # credit from https://stackoverflow.com/a/61693402/10732321
        get_uni_chars_cache.ranges = [
            (0x0021, 0x02FF),  # Basic Latin
            (0x0080, 0x00FF),  # Latin-1 Supplement
            (0x0100, 0x017F),  # Latin Extended A
            (0x0180, 0x024F),  # Latin Extended B
            (0x0250, 0x02AF),  # IPA Extensions
            (0x1D00, 0x1D7F),  # Phonetic Extensions
            (0x1D80, 0x1DBF),  # Phonetic Extensions Supplement
            (0x1E00, 0x1EFF),  # Latin Extended Additional
            (0x2100, 0x214F)  # Letter-like Symbols
        ]
    if not hasattr(get_uni_chars_cache, 'omit'):
        get_uni_chars_cache.omit = {  # Characters that can't be printed
            0x7f, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90,
            0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f, 0xa0, 0xad
        }
    it = chain_its((range(*r) for r in get_uni_chars_cache.ranges))
    chars = {chr(i) for i in it if i not in get_uni_chars_cache.omit}
    return sorted(chars)


class Score2Chars:
    """
    To fit to existing WordPiece training, mapping between
        1) my music `score` format and 2) sequence of contiguous characters
    """
    uni_chars_cache = get_uni_chars_cache()

    def __init__(
            self, vocab: MusicVocabulary, chars: List[str] = None, continuing_prefix: str = '##',
            independent_global_token: bool = False, punctuate: bool = False, omit_eos: bool = False
    ):
        """
        :param vocab: Handles music vocabulary processing, such as mapping from token to ordinal/id
        :param chars: A list of characters
            Intended for mapping each ordinal
        :param independent_global_token: If True, global metadata token are not merged
            i.e. Time Signature, Tempo, Key
            Analogous to maintaining the character
        :param punctuate: If True, WordPiece merging stops at bar separation, melody and bass prefixes, and tuplets
            Analogous to punctuation
        """
        self.vocab = vocab
        if chars:
            assert len(chars) == len(vocab)
        else:
            chars = Score2Chars.get_uni_chars(len(vocab))
        assert all(c != ' ' for c in chars)
        self.dec_chars = chars
        self.enc_chars = {c: i for i, c in enumerate(chars)}
        self.continuing_prefix = continuing_prefix

        self.independent_global_token = independent_global_token
        self.punctuate = punctuate
        self.need_split = independent_global_token or punctuate
        self.spec_toks = {
            self.vocab.start_of_bar, self.vocab.start_of_tuplet, self.vocab.end_of_tuplet, self.vocab.end_of_song,
            self.vocab.start_of_melody, self.vocab.start_of_bass
        }
        self.omit_eos = omit_eos

    @staticmethod
    def get_uni_chars(n: int) -> List[str]:
        assert 0 < n <= len(Score2Chars.uni_chars_cache)
        return Score2Chars.uni_chars_cache[:n]

    def __call__(self, score: str, clean: bool = True) -> str:
        """
        Vanilla music token representation => character string ready for training
        """
        if self.need_split:  # prevent merge via separating by space, see `__call__` pre_tokenizer
            toks = self.split(score, join=False)  # efficient as `encode` can take split tokens

            # sanity_check = False
            sanity_check = True
            if sanity_check:
                ret = ' '.join([self.encode(t) for t in toks])
                _toks = ret.split()
                _toks = [self.decode(_t) for _t in _toks]
                assert ' '.join(_toks) == self.vocab.clean_uncommon(score)
                for t in _toks:
                    mic(t)
                exit(1)
            return ' '.join([self.encode(t) for t in toks])
        else:
            return self.encode(score)

    def split(self, score: str, join: bool = True) -> Union[List[str], List[List[str]]]:
        toks = score.split()
        if self.need_split:
            ts, tp, key, toks = toks[0], toks[1], None, toks[2:]
            assert self.vocab.type(ts) == VocabType.time_sig
            assert self.vocab.type(tp) == VocabType.tempo

            t1 = self.vocab.type(toks[0])
            assert t1 in (VocabType.special, VocabType.key)
            if t1 == VocabType.key:
                key, toks = toks[0], toks[1:]
            assert toks[0] == self.vocab.start_of_bar
            if not self.omit_eos:
                assert toks[-1] == self.vocab.end_of_song
            if self.independent_global_token:
                if join:
                    words = [[ts], [tp]]
                else:
                    words = [ts, tp]
                if key:
                    words.append([key] if join else key)
                if self.punctuate:
                    words += self._split_bar_notes(toks, join=join)
                else:
                    words += ' '.join(toks)
                return words
            else:  # punctuate is True
                word_1st = f'{ts} {tp}'
                if not join:
                    word_1st = [word_1st]
                return [word_1st] + self._split_bar_notes(toks, join=join)
        else:
            return toks

    def _split_bar_notes(self, toks: List[str], join: bool = True) -> Union[List[str], List[List[str]]]:
        words, curr_word = [], []  # word as in how it will look like after pre-tokenization
        for tok in toks:
            if tok in self.spec_toks:
                words.append(curr_word)
                words.append([tok])
                curr_word = []
            else:
                curr_word.append(tok)
        if curr_word:
            words.append(curr_word)
        return [' '.join(w) for w in words] if join else words

    def encode(self, s: Union[str, List[str]], clean: bool = True) -> str:
        """
        score => chars
        """
        toks = s.split() if isinstance(s, str) else s
        if clean:
            toks = [self.vocab.clean_uncommon_token(t) for t in toks]
        mic(toks)
        mic([self.vocab.tok2id[tok] for tok in toks])
        mic(len(self.dec_chars), len(self.vocab))
        return ''.join([self.dec_chars[self.vocab.tok2id[tok]] for tok in toks])

    def decode(self, s: str) -> str:
        """
        chars => score
        """
        def _decode(s_: str) -> str:
            return ' '.join([self.vocab.id2tok[self.enc_chars[c]] for c in s_])
        return ' '.join([_decode(s) for s in s.split()]) if self.need_split else _decode(s)

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
    continuing_prefix = WORDPIECE_CONTINUING_PREFIX

    def __init__(self, vocab: MusicVocabulary, augment_key: bool = True, **kwargs):
        """
        :param vocab: Music Vocabulary, for internal mapping between music tokens & characters
        :param augment_key: If true, each score have each possible key inserted & pitch shifted accordingly
            Otherwise, train with raw music extraction scores
        """
        # vocab matters cos it influences base vocab
        if augment_key:
            assert vocab.pitch_kind == 'degree'
        else:  # TODO: maybe support `step` vocab?
            assert vocab.pitch_kind == 'midi'
        self.vocab = vocab
        self.augment_key = augment_key

        self.s2c = Score2Chars(vocab=vocab, continuing_prefix=WordPieceMusicTrainer.continuing_prefix, **kwargs)

    def __call__(self, vocab_size: int = 2**13, songs: List[Dict[str, Any]] = None, save: Union[bool, str] = None):
        # TODO: don't need to pass `vocab` to `WordPiece`?
        # every token should be known
        # TODO: What is `max_input_chars_per_word`? set no lim
        logger = get_logger(self.__class__.__qualname__)
        d_log = {'vocab-size': vocab_size, '#song': len(songs)}
        logger.info(f'Training launched with {pl.i(d_log)}')

        tokenizer = Tokenizer(model=models.WordPiece(vocab=None, max_input_chars_per_word=int(1e10)))
        if self.s2c.need_split:
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.decoder = decoders.WordPiece(prefix=WordPieceMusicTrainer.continuing_prefix)
        init_alph = self.s2c.dec_chars
        assert len(init_alph) <= vocab_size
        trainer = WordPieceTrainer(
            vocab_size=vocab_size, initial_alphabet=init_alph, show_progress=True,
            continuing_subword_prefix=WordPieceMusicTrainer.continuing_prefix
        )
        if self.augment_key:
            ki = KeyInsert(vocab=self.vocab)  # pitch kind irrelevant
            ps = PitchShift(vocab_degree=self.vocab)  # pitch kind irrelevant

            def gen():
                for s in songs:
                    txt = s['score']
                    for key in s['keys']:
                        _txt = ki(text=txt, key=key)
                        yield ps(text=_txt)
            gen = gen()
        else:
            gen = (self.s2c(s['score']) for s in songs)
        tokenizer.train_from_iterator(gen, trainer=trainer)
        if save:
            fnm = save if isinstance(save, str) else 'WordPiece-Tokenizer'
            date = now(fmt='short-date')
            fnm = f'{date}_{fnm}_{{vsz={vocab_size}, n={len(songs)}}}'
            path_tok = os_join(u.tokenizer_path, f'{fnm}.json')
            tokenizer.save(path_tok)
            logger.info(f'{pl.i("Tokenizer")} saved to {pl.i(path_tok)}')
            path_meta = os_join(u.tokenizer_path, f'{fnm}_meta.json')
            with open(path_meta, 'w') as f:
                json.dump(dict(  # For reconstructing class properties, see `WordPieceMusicTokenizer`
                    music_vocab=dict(prec=self.vocab.precision),
                    score2chars=dict(
                        chars=self.s2c.dec_chars,
                        independent_global_token=self.s2c.independent_global_token,
                        punctuate=self.s2c.punctuate
                    ),
                ), f, indent=4)
            logger.info(f'{pl.i("Tokenizer")} music metadata saved to {pl.i(path_meta)}')
        return tokenizer

    def music_vocab(self, tokenizer: Tokenizer) -> Dict[str, int]:
        return self.s2c.char_vocab2vocab(tokenizer.get_vocab())


class MyPreTrainedTokenizerFast(PreTrainedTokenizerFast):
    """
    Override tokenization return vars
    """
    model_input_names = ['input_ids']


class WordPieceMusicTokenizer(MusicTokenizer):

    def __init__(
            self, tokenizer: Tokenizer, precision: int = 5, s2c_args: Dict = None, omit_eos: bool = False, **kwargs
    ):
        """
        :param tokenizer: A trained WordPiece tokenizer on characters
        """
        super().__init__(precision=precision, name_or_path=self.__class__.__qualname__, **kwargs)
        self._tokenizer = MyPreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token=self.pad_token, eos_token=self.eos_token
        )  # now vocab size is correctly set
        self.continuing_prefix = tokenizer.decoder.prefix
        self.omit_eos = omit_eos
        self.s2c = Score2Chars(
            vocab=self.vocab, continuing_prefix=self.continuing_prefix, omit_eos=omit_eos, **s2c_args
        )

        # self._add_special_token(self.vocab.pad)
        assert self._tokenizer.pad_token_id is None  # TODO: Unlike `MusicTokenizer`, not sure why not defined already
        self._tokenizer.pad_token_id = tokenizer.token_to_id(self.s2c.encode(self.pad_token))

        self._id2pchs: Dict[int, List[int]] = dict()  # cache, from each id to pitch if it contains any
        for i in range(self.vocab_size):
            # note the same token in vanilla tokenizer, may appear twice,
            #   once for being part of base vocab, another time as part of WordPiece continuation subword
            toks = self._convert_id_to_token(i).split()
            self._id2pchs[i] = super().ids2pitches(toks)

    @classmethod
    def from_file(cls, fnm: str, output_path: str = u.tokenizer_path, **kwargs):
        _tokenizer = Tokenizer.from_file(os_join(output_path, f'{fnm}.json'))
        with open(os_join(output_path, f'{fnm}_music_meta.json'), 'r') as f:
            meta = json.load(f)
        prec = meta['music_vocab'].pop('prec')
        return cls(_tokenizer, precision=prec, s2c_args=meta['score2chars'], **kwargs)

    @property
    def model_max_length(self) -> int:
        return self._model_max_length

    @model_max_length.setter
    def model_max_length(self, value: int):
        if hasattr(self, '_tokenizer'):  # so that no error when parent __init__ runs
            self._tokenizer.model_max_length = value
        self._model_max_length = value

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return self._tokenizer(self.s2c(text, clean=True), **kwargs)
        else:
            assert isinstance(text, (list, tuple)) and isinstance(text[0], str)
            return self._tokenizer([self.s2c(t, clean=True) for t in text], **kwargs)

    def tokenize(self, text, mode: str = 'music', entire_score: bool = True, **kwargs):
        """
        :param text: Music score
        :param mode: One of [`music`, `char`], whether to return in human-readable representation
        :param entire_score: If true, enforce check for entire score passed in
            Otherwise, the tokens are not split at all
                For this reason, **use with caution**, Intended for debugging, see `check_trained_has_single_token`
        """
        ca.check_mismatch('Tokenization Mode', mode, ['music', 'char'])
        if isinstance(text, str):
            encoded = self.s2c(text, clean=True) if entire_score else self.s2c.encode(text, clean=True)
            toks = self._tokenizer.tokenize(encoded, **kwargs)
            return [self.s2c.trained_tok2tok(t) for t in toks] if mode == 'music' else toks
        else:
            raise NotImplementedError('Not implemented for iterable input')

    def encode(self, text, **kwargs):
        if isinstance(text, str):
            return self._tokenizer.encode(self.s2c(text, clean=True), **kwargs)
        else:
            raise NotImplementedError('TODO')

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

    def ids2pitches(self, ids: Iterable[int]) -> List[int]:
        return sum([self._id2pchs[int(i)] for i in ids], start=[])


def load_trained_tokenizer(  # has independent global token & bar split
        fnm: str = None, pitch_kind: str = None, **kwargs
) -> WordPieceMusicTokenizer:
    if pitch_kind is None:
        pitch_kind = 'midi'
    if pitch_kind == 'midi':
        fnm = fnm or '22-10-03_WordPiece-Tokenizer_{dnm=all}_{vsz=16384, n=178825}'
    elif pitch_kind == 'step':
        raise NotImplementedError('Recompute tokenizer w/ new music extraction tokens but no key aug')
    else:
        assert pitch_kind == 'degree'
        raise NotImplementedError('TODO')
    return WordPieceMusicTokenizer.from_file(fnm, is_wordpiece=True, pitch_kind=pitch_kind, **kwargs)


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
    from tqdm.auto import tqdm

    from musicnlp._sample_score import sample_full_midi, sample_full_step
    from musicnlp.preprocess import load_songs, DATASET_NAME2MODE2FILENAME

    def sanity_check_split():
        pre_tokenizer = pre_tokenizers.WhitespaceSplit()  # split on whitespace only
        mic(pre_tokenizer.pre_tokenize_str(sample_full))
    # sanity_check_split()

    # md = 'melody'
    md = 'full'
    pop, mst, lmd = [get(DATASET_NAME2MODE2FILENAME, f'{dnm}.{md}') for dnm in ['POP909', 'MAESTRO', 'LMD']]

    def check_tokenize_train():
        mv = MusicVocabulary()
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

    def check_s2c_cache_chars():
        """
        see at a glance all the characters & remove ones that can't be displayed

        ideally, those should take same width as English characters
        """
        cache = Score2Chars.uni_chars_cache
        mic(len(cache))
        print(''.join(cache))
        not_printable = []
        for i, c in enumerate(cache):
            # mic(i, c)
            if not c.isprintable():
                mic(i, c)
                not_printable.append(c)
        mic(not_printable)
    # check_s2c_cache_chars()

    def try_char_map(kind: str = 'midi'):
        mv = MusicVocabulary(pitch_kind=kind)
        if kind == 'midi':
            sample = sample_full_midi
        else:
            assert kind == 'step'
            sample = sample_full_step
        # chs = get_uni_chars(40)
        # mic(chs, len(chs))
        # exit(1)
        s2c = Score2Chars(mv)
        sample_txt_ = mv.clean_uncommon(sample, return_joined=False)
        encoded = s2c(sample_txt_)
        mic(encoded)
        decoded = s2c.decode(encoded)
        mic(decoded)
        assert ' '.join(sample_txt_) == decoded
    # try_char_map(kind='midi')
    try_char_map(kind='step')

    def train():
        from collections import Counter

        from tqdm.auto import tqdm

        dnms = [pop]
        # dnms = [pop, mst, lmd]

        aug_key = True
        mv = MusicVocabulary(pitch_kind='degree' if aug_key else 'midi')

        vocab_size, svs = None, None
        sv = True
        # sv = False
        if len(dnms) == 1:
            vocab_size = 4096
            if sv:
                sv = 'WordPiece-Tokenizer_{dnm=POP909}'
        elif len(dnms) == 2:
            vocab_size = 4096 * 2
            if sv:
                sv = 'WordPiece-Tokenizer_{dnm=POP&MST}'
        elif len(dnms) == 3:
            vocab_size = 4096 * 4
            if sv:
                sv = 'WordPiece-Tokenizer_{dnm=all}'
        wmt = WordPieceMusicTrainer(vocab=mv, augment_key=aug_key, independent_global_token=True, punctuate=True)
        songs = load_songs(*dnms, score_only=False)
        tokenizer = wmt(vocab_size=vocab_size, songs=songs, save=sv)

        s2c = wmt.s2c

        check_preserve = False
        # check_preserve = True
        if check_preserve:
            sample_txt_ = mv.clean_uncommon(sample_full)
            encoded = s2c(sample_txt_)
            encoded = tokenizer.encode(encoded).ids
            decoded = tokenizer.decode(encoded)
            decoded = s2c.decode(decoded)
            assert decoded == sample_txt_
            exit(1)

        check_dist = False
        # check_dist = True
        if check_dist:
            c = Counter()
            it = tqdm(songs, desc='Counting token dist', unit='song')
            for song in it:
                toks = mv.clean_uncommon(song, return_joined=False)
                it.set_postfix(n_tok=len(toks))
                song = ' '.join(toks)
                encoded = s2c(song)
                encoded = tokenizer.encode(encoded)
                c.update([s2c.trained_tok2tok(t) for t in encoded.tokens])
            mic(c)
    # train()

    def check_trained_property():
        # fnm = '2022-06-15_20-50-15_Word-Piece-Music-Tokenizer, dnm=POP909, vsz=4096, n=909'
        fnm = '2022-06-15_21-41-08_Word-Piece-Music-Tokenizer, dnm=all, vsz=16384, n=178825'
        tokenizer = load_trained_tokenizer(fnm)
        # mic(tokenizer)

        # map_single = _CheckTrainedMap(mv, tokenizer)
        # mic(map_single(sample_txt2))

        sample_txt2_cleaned = tokenizer.vocab.clean_uncommon(sample_full)
        # encoded = tokenizer.tokenize(sample_txt2_cleaned)
        # mic(encoded)

        inputs = tokenizer(sample_txt2_cleaned, padding=True)
    # check_trained_property()

    def check_trained_has_single_token():
        """
        Check each single token in vanilla vocab can be encoded into single token, 
        i.e. trained vocab has each single token, so that no `UNK` token needed
        """
        # fnm = '2022-06-15_21-21-16_Word-Piece-Music-Tokenizer, dnm=POP909, vsz=4096, n=909'
        fnm = '2022-06-15_21-41-08_Word-Piece-Music-Tokenizer, dnm=all, vsz=16384, n=178825'
        tokenizer = load_trained_tokenizer(fnm)
        vocab = tokenizer.vocab
        it = tqdm(vocab.tok2id.keys())
        for tok in it:
            encoded = tokenizer.encode(tok, entire_score=False)
            it.set_postfix(dict(tok=tok, encoded=encoded))
            if len(encoded) != 1:
                mic(tok, encoded)
                mic(tokenizer.s2c.encode(tok))
            assert len(encoded) == 1
    # check_trained_has_single_token()

    def check_trained_tokenize_all():
        from collections import Counter

        aug_key = True
        mv = MusicVocabulary(pitch_kind='degree' if aug_key else 'midi')

        fnm = '22-10-03_WordPiece-Tokenizer_{dnm=all}_{vsz=16384, n=178825}'
        tokenizer = WordPieceMusicTokenizer.from_file(fnm)
        # inputs = tokenizer(sample_txt)
        # mic(tokenizer.decode(inputs['input_ids']))

        check_preserve = True  # sanity check, encoding & decoding, every token is still preserved
        check_dist = True

        # dnms = [pop]
        # dnms = [lmd]
        dnms = [pop, mst, lmd]
        songs = load_songs(*dnms)
        concurrent = True
        # concurrent = False
        if concurrent:
            map_single = _CheckTrainedMap(mv, tokenizer)
            lst_ids = conc_map(
                map_single, songs, with_tqdm=dict(desc='Checking trained tokenizer', chunksize=128), mode='process'
            )
            toks = conc_map(
                tokenizer.convert_ids_to_tokens, lst_ids,
                with_tqdm=dict(desc='ids=>tokens', chunksize=128), mode='process'
            )
            c = Counter()
            for toks in tqdm(toks, desc='Updating token dist', unit='song'):
                c.update(toks)
            mic(c)
            with open(os_join(u.tokenizer_path, f'{fnm} distribution check.json'), 'w') as f:
                json.dump(dict(count=c), f, indent=4)
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
    # check_trained_tokenize_all()

    def check_id2pch():
        tokenizer = MusicTokenizer()
        ids = tokenizer.encode(sample_full)
        pchs = tokenizer.ids2pitches(ids)
        mic(len(ids), len(pchs))

        wp_tokenizer = load_trained_tokenizer()
        ids = wp_tokenizer.encode(sample_full)
        wp_pchs = wp_tokenizer.ids2pitches(ids)
        mic(len(ids), len(wp_pchs))
        assert wp_pchs == pchs
    # check_id2pch()