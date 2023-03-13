"""
Combine individual music token in `music_vocab`, treating each token as a character as in WordPiece tokenizer training
    i.e. Base vocab is tokens

Intended to trade sequence length with vocabulary size
    The vanilla tokenizer takes up
"""

import json
from os.path import join as os_join
from typing import List, Tuple, Dict, Union, Iterable, Any

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers import pre_tokenizers, models, decoders

from stefutil import *
from musicnlp.util import *
from musicnlp.vocab import MusicVocabulary, VocabType, WORDPIECE_CONTINUING_PREFIX, MusicTokenizer
from musicnlp.preprocess import transform, dataset


__all__ = ['WordPieceMusicTrainer', 'WordPieceMusicTokenizer', 'load_wordpiece_tokenizer']


logger = get_logger('WordPiece Tokenizer')


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
            # mic(len(chars), len(vocab), vocab.pitch_kind)
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

    def __call__(self, score: Union[str, List[str]]) -> str:
        """
        Vanilla music token representation => character string ready for training
        """
        if self.need_split:  # prevent merge via separating by space, see `__call__` pre_tokenizer
            toks = self.split(score, join=False)  # efficient as `encode` can take split tokens

            sanity_check = False
            # sanity_check = True
            if sanity_check:  # encode-decode reconstructs the original sequence
                ret = ' '.join([self.encode_single(t) for t in toks])
                _toks = ret.split()
                _toks = [self.decode(_t) for _t in _toks]

                ori = score if isinstance(score, str) else ' '.join(score)
                new = ' '.join(_toks)
                assert ori == new
                raise NotImplementedError
            # mic(toks)
            # mic([self.encode_single(t) for t in toks])
            return ' '.join([self.encode_single(t) for t in toks])
        else:
            return self.encode_single(score)

    def encode(self, score: str) -> str:
        return self.__call__(score)

    def split(self, score: Union[str, List[str]], join: bool = True) -> Union[List[str], List[List[str]]]:
        toks = score.split() if isinstance(score, str) else score
        if self.need_split:
            ts, tp, key, omit, toks = toks[0], toks[1], None, None, toks[2:]
            assert self.vocab.type(ts) == VocabType.time_sig
            assert self.vocab.type(tp) == VocabType.tempo

            t1 = self.vocab.type(toks[0])
            assert t1 in (VocabType.special, VocabType.key)
            if t1 == VocabType.key:
                key, toks = toks[0], toks[1:]

            t1 = toks[0]
            if t1 == self.vocab.omitted_segment:
                omit, toks = toks[0], toks[1:]

            assert toks[0] == self.vocab.start_of_bar
            if not self.omit_eos:
                assert toks[-1] == self.vocab.end_of_song
            if self.independent_global_token:
                if join:
                    words = [ts, tp]
                else:
                    words = [[ts], [tp]]

                if key:
                    words.append([key] if join else key)
                if omit:
                    words.append([omit] if join else omit)

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
                if curr_word:
                    words.append(curr_word)
                words.append([tok])
                curr_word = []
            else:
                curr_word.append(tok)
        if curr_word:
            words.append(curr_word)
        return [' '.join(w) for w in words] if join else words

    def encode_single(self, s: Union[str, List[str]]) -> str:
        """
        score => chars
        """
        toks = s.split() if isinstance(s, str) else s
        # mic('in encode single', s)
        return ''.join([self.dec_chars[self.vocab.tok2id[tok]] for tok in toks])

    def decode(self, s: str) -> str:
        """
        chars => score
        """
        return ' '.join([self.decode_single(s) for s in s.split()]) if self.need_split else self.decode_single(s)

    def decode_single(self, s_: str) -> str:
        return ' '.join([self.vocab.id2tok[self.enc_chars[c]] for c in s_])

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

    def __init__(self, vocab: MusicVocabulary, pitch_kind: str = None, augment_key: bool = True, **kwargs):
        """
        :param vocab: Music Vocabulary, for internal mapping between music tokens & characters
            Determines base vocabulary for tokenizer
        :param augment_key: If true, each score have each possible key inserted & pitch shifted accordingly
            Otherwise, train with raw music extraction scores
        """
        if vocab:
            if pitch_kind:
                assert pitch_kind == vocab.pitch_kind
            self.vocab = vocab
        else:
            self.vocab = MusicVocabulary(pitch_kind=pitch_kind)
        self.pitch_kind = self.vocab.pitch_kind

        # they must go hand-in-hand
        assert (self.pitch_kind == 'degree' and augment_key) or (self.pitch_kind != 'degree' and not augment_key)
        self.augment_key = augment_key

        self.s2c = Score2Chars(vocab=vocab, continuing_prefix=WordPieceMusicTrainer.continuing_prefix, **kwargs)

    def __call__(
            self, vocab_size: int = 2**13, songs: List[Dict[str, Any]] = None,
            concurrent: Union[bool, int] = False, save: Union[bool, str] = None
    ):
        # TODO: don't need to pass `vocab` to `WordPiece`?
        # every token should be known
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

        d_log = {'vocab-size': vocab_size, '#song': len(songs)}

        # since input text will be in `step`
        sr_vocab = self.vocab if self.vocab.pitch_kind == 'step' else MusicVocabulary(pitch_kind='step')
        sr = transform.SanitizeRare(vocab=sr_vocab, return_as_list=True, for_midi=self.pitch_kind == 'midi')
        if self.augment_key:
            assert self.pitch_kind == 'degree'

            out = dataset.iter_song_w_all_keys(songs)
            it, n = out.generator, out.total
            d_log['#key-augmented-song'] = out.total

            it = ((sr(txt), key) for txt, key in it)

            fn = transform.AugmentKey(vocab=self.vocab, return_as_list=True)
            bsz = None
            if concurrent:
                bsz = 32 if isinstance(concurrent, bool) else concurrent
                it = conc_yield(fn=fn, args=it, mode='process', batch_size=bsz)
            else:
                it = (fn(pair) for pair in it)
            d_log['concurrent'] = concurrent
            if concurrent:
                d_log['batch_size'] = bsz
        else:  # `midi`, `step`
            it = (s['score'] for s in songs)
            it = (sr(txt) for txt in it)
            if self.pitch_kind == 'midi':  # Since expect extracted song sequences to be in pitch_kind `step`
                mv = MusicVocabulary(pitch_kind='step', is_wordpiece=True)
                tmp = transform.ToMidiPitch(vocab=mv, return_as_list=True)
                it = (tmp(s) for s in it)
        
        sanity_check = False
        # sanity_check = True
        mic(sanity_check)
        if sanity_check:
            for e in tqdm(it, total=len(songs), desc='Sanity check s2c'):
                # mic(e[:200])
                # raise NotImplementedError
                try:
                    self.s2c(e)
                except Exception as exp:
                    mic(e)
                    mic(exp)
                    raise exp
            raise NotImplementedError

        logger.info(f'Training WordPiece tokenization w/ {pl.i(d_log)}')
        gen = (self.s2c(s) for s in it)
        tokenizer.train_from_iterator(gen, trainer=trainer)
        if save:
            fnm = save if isinstance(save, str) else 'WordPiece-Tokenizer'
            date = now(fmt='short-date')
            meta = dict(vsz=vocab_size, n=len(songs), pch=self.pitch_kind[0])
            if self.augment_key:
                meta['aug-key'] = 'T'
            fnm = f'{date}_{fnm}_{pl.pa(meta)}'
            path_tok = os_join(u.tokenizer_path, f'{fnm}.json')
            tokenizer.save(path_tok)
            logger.info(f'Tokenizer saved to {pl.i(path_tok)}')
            path_meta = os_join(u.tokenizer_path, f'{fnm}_meta.json')
            with open(path_meta, 'w') as f:
                json.dump(dict(  # For reconstructing class properties, see `WordPieceMusicTokenizer`
                    music_vocab=dict(precison=self.vocab.precision, pitch_kind=self.vocab.pitch_kind),
                    score2chars=dict(
                        chars=self.s2c.dec_chars,
                        independent_global_token=self.s2c.independent_global_token,
                        punctuate=self.s2c.punctuate
                    ),
                    # For sanity check vocabulary not obsolete, see `WordPieceMusicTokenizer::from_file`
                    tok2id=self.vocab.tok2id
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
        init_args = dict(precision=precision, name_or_path=self.__class__.__qualname__, is_wordpiece=True) | kwargs
        super().__init__(**init_args)
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
        self._tokenizer.pad_token_id = tokenizer.token_to_id(self.s2c.encode_single(self.pad_token))

        self._id2pchs_inc: Dict[int, List[int]] = dict()  # cache, from each id to pitch if it contains any
        self._id2pchs_exc: Dict[int, List[int]] = dict()
        for i in range(self.vocab_size):
            # note the same token in vanilla tokenizer, may appear twice,
            #   once for being part of base vocab, another time as part of WordPiece continuation subword
            toks = self._convert_id_to_token(i).split()
            self._id2pchs_inc[i] = super().ids2pitches(toks, include_rest_pitch=True)
            self._id2pchs_exc[i] = super().ids2pitches(toks, include_rest_pitch=False)

    @classmethod
    def from_file(cls, fnm: str, output_path: str = u.tokenizer_path, **kwargs):
        path = os_join(output_path, fnm)
        logger.info(f'Loading wordpiece tokenizer from {pl.i(path)}... ')
        _tokenizer = Tokenizer.from_file(f'{path}.json')
        with open(f'{path}_meta.json', 'r') as f:
            meta = json.load(f)
        init_args = meta['music_vocab'] | kwargs
        ret = cls(_tokenizer, s2c_args=meta['score2chars'], **init_args)

        t2i = meta.get('tok2id', None)   # backward compatibility
        if t2i:
            assert ret.vocab.tok2id == t2i
        return ret

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
            return self._tokenizer(self.s2c(text), **kwargs)
        else:
            assert isinstance(text, (list, tuple)) and isinstance(text[0], str)
            return self._tokenizer([self.s2c(t) for t in text], **kwargs)

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
            encoded = self.s2c(text) if entire_score else self.s2c.encode_single(text)
            toks = self._tokenizer.tokenize(encoded, **kwargs)
            return [self.s2c.trained_tok2tok(t) for t in toks] if mode == 'music' else toks
        else:
            raise NotImplementedError('Not implemented for iterable input')

    def encode(self, text, entire_score: bool = True, **kwargs):
        if isinstance(text, str):
            encoded = self.s2c(text) if entire_score else self.s2c.encode_single(text)
            return self._tokenizer.encode(encoded, **kwargs)
        else:
            raise NotImplementedError('TODO')

    def decode(self, token_ids, **kwargs):
        decoded = self._tokenizer.decode(token_ids, **kwargs, clean_up_tokenization_spaces=False)
        if isinstance(decoded, str):
            return self.s2c.decode(decoded)
        else:
            raise NotImplementedError('Not implemented for iterable input')

    def _convert_id_to_token(self, index: int) -> str:
        return self.s2c.decode(self._tokenizer.decode(index).removeprefix(self.continuing_prefix))

    def ids2pitches(self, ids: Iterable[int], include_rest_pitch: bool = True) -> List[int]:
        i2p = self._id2pchs_inc if include_rest_pitch else self._id2pchs_exc
        return sum([i2p[int(i)] for i in ids], start=[])


def load_wordpiece_tokenizer(  # has independent global token & bar split
        fnm: str = None, pitch_kind: str = None, **kwargs
) -> WordPieceMusicTokenizer:
    pitch_kind = pitch_kind or 'midi'
    if pitch_kind == 'midi':
        # Obsolete for no [OMIT] token; TODO Re-run, maybe w/ a larger vocab size
        # fnm = fnm or '22-10-03_WordPiece-Tokenizer_{dnm=all}_{vsz=16384, n=178825}'
        fnm = fnm or '22-11-13_WordPiece-Tokenizer_{dnm=all}_{vsz=32768, n=178825, pch=m}'
    elif pitch_kind == 'step':
        fnm = fnm or '22-10-25_WordPiece-Tokenizer_{dnm=POP&MST}_{vsz=8192, n=2185, pch=s}'
    else:
        assert pitch_kind == 'degree'
        fnm = fnm or '22-10-26_WordPiece-Tokenizer_{dnm=all}_{vsz=32768, n=178825, pch=d, aug-key=T}'
    return WordPieceMusicTokenizer.from_file(fnm, is_wordpiece=True, pitch_kind=pitch_kind, **kwargs)


class _CheckTrainedSingle:
    """
    **debugging**, see `check_trained`
    """
    def __init__(
            self, tokenizer: WordPieceMusicTokenizer = None, augment_key: bool = False, check_reconstruct: bool = False
    ):
        self.tokenizer = tokenizer
        vocab = self.tokenizer.vocab
        self.pitch_kind = vocab.pitch_kind
        sr_vocab = vocab if self.pitch_kind == 'step' else MusicVocabulary(pitch_kind='step')
        self.sr = transform.SanitizeRare(vocab=sr_vocab, for_midi=self.pitch_kind == 'midi', return_as_list=True)
        self.tmp = None
        if self.pitch_kind == 'midi':
            self.tmp = transform.ToMidiPitch(vocab=sr_vocab, return_as_list=True)

        self.augment_key, self.ak = augment_key, None
        if augment_key:
            self.ak = transform.AugmentKey(vocab=vocab)

        self.check_reconstruct = check_reconstruct

    def __call__(self, text: Union[str, Tuple[str, str]] = None) -> List[str]:
        if self.augment_key:
            assert isinstance(text, tuple)
            text, key = text
            text = self.sr(text)
            text = self.ak((text, key))
        else:
            assert isinstance(text, str)
            text = self.sr(text)

        if self.pitch_kind == 'midi':
            text = self.tmp(text)

        toks = text if isinstance(text, list) else text.split()
        assert all(t in self.tokenizer.vocab for t in toks)
        text = ' '.join(toks)
        ids = self.tokenizer(text).input_ids
        toks = self.tokenizer.convert_ids_to_tokens(ids)

        sanity_check = False
        # sanity_check = True
        if sanity_check:
            ori = text[:200]
            new = ' '.join(toks)[:200]
            mic(ori, new)
            raise NotImplementedError

        if self.check_reconstruct:
            dec = self.tokenizer.decode(ids)
            assert dec == text
        return toks


if __name__ == '__main__':
    from collections import Counter

    from tqdm.auto import tqdm

    from musicnlp._sample_score import sample_full_midi, sample_full_step

    mic.output_width = 256

    def sanity_check_split():
        pre_tokenizer = pre_tokenizers.WhitespaceSplit()  # split on whitespace only
        mic(pre_tokenizer.pre_tokenize_str(sample_full_midi))
    # sanity_check_split()

    # md = 'melody'
    md = 'full'
    pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD')

    def check_tokenize_train():
        mv = MusicVocabulary(is_wordpiece=True)
        unk = '[UNK]'
        tokenizer = Tokenizer(model=models.WordPiece(vocab=None, unk_token=unk, max_input_chars_per_word=int(1e10)))
        # input scores already cleaned, no normalizer needed
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = WordPieceTrainer(
            vocab_size=int(2e5), initial_alphabet=list(mv.tok2id.keys()), special_tokens=[unk], show_progress=True
        )
        songs = dataset.load_songs(pop)
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
        mv = MusicVocabulary(pitch_kind=kind, is_wordpiece=True)
        if kind == 'midi':
            sample = sample_full_midi
        else:  # `step`, `degree`
            sample = sample_full_step
            if kind == 'degree':
                ki = transform.KeyInsert(vocab=mv, return_as_list=True)  # pitch kind irrelevant
                ps = transform.PitchShift(vocab_degree=mv)  # pitch kind irrelevant
                key = 'CMajor'  # Pick an arbitrary key
                sample = ps(text=ki(text=sample, key=key))
                # mic(sample)
        # chs = get_uni_chars(40)
        # mic(chs, len(chs))
        # exit(1)
        s2c = Score2Chars(mv)
        sample_txt_ = mv.sanitize_rare_tokens(sample, return_as_list=False)
        encoded = s2c(sample_txt_)
        mic(encoded)
        decoded = s2c.decode(encoded)
        mic(decoded)
        assert ' '.join(sample_txt_) == decoded
    # try_char_map(kind='midi')
    # try_char_map(kind='step')
    # try_char_map(kind='degree')

    def profile_s2c():
        vocab = MusicVocabulary(pitch_kind='step', is_wordpiece=True)
        songs = dataset.load_songs(pop)
        s2c = Score2Chars(vocab=vocab, independent_global_token=True, punctuate=True)

        def fn():
            for song in tqdm(songs):
                s2c(song)
        profile_runtime(fn)
    # profile_s2c()

    def train():
        # dnms = [pop]
        # dnms = [pop, mst]
        dnms = [pop, mst, lmd]
        mic(dnms)

        # pch_kd = 'midi'
        # pch_kd = 'step'
        pch_kd = 'degree'
        aug_key = pch_kd == 'degree'
        mic(pch_kd, aug_key)
        mv = MusicVocabulary(pitch_kind=pch_kd, is_wordpiece=True)

        # conc = 128 if len(dnms) == 3 else 32
        conc = False
        mic(conc)

        vocab_size, svs = None, None
        sv = True
        # sv = False
        if len(dnms) == 1:
            vocab_size = 4096
            if sv:
                sv = 'WordPiece-Tokenizer_{dnm=POP909}'
        elif len(dnms) == 2:
            # vocab_size = 4096 * 2
            vocab_size = 4096 * 4
            if sv:
                sv = 'WordPiece-Tokenizer_{dnm=POP&MST}'
        elif len(dnms) == 3:
            # vocab_size = 4096 * 4
            # vocab_size = 4096 * 8
            vocab_size = 4096 * 8 * 8
            if sv:
                sv = 'WordPiece-Tokenizer_{dnm=all}'
        wmt = WordPieceMusicTrainer(
            vocab=mv, pitch_kind=pch_kd, augment_key=aug_key, independent_global_token=True, punctuate=True
        )
        songs = dataset.load_songs(*dnms)
        wmt(vocab_size=vocab_size, songs=songs, save=sv, concurrent=conc)
    # train()

    def check_trained_property():
        aug_key = True
        # aug_key = False
        if aug_key:
            pch_kd = 'degree'
            fnm = '22-10-25_WordPiece-Tokenizer_{dnm=POP&MST}_{vsz=16384, n=2185, pch=d, aug-key=T}'
        else:
            pch_kd = 'step'
            fnm = '22-10-25_WordPiece-Tokenizer_{dnm=POP&MST}_{vsz=8192, n=2185, pch=s}'
        tokenizer = load_trained_tokenizer(fnm, pitch_kind=pch_kd)

        if aug_key:
            map_single = _CheckTrainedSingle(tokenizer=tokenizer, augment_key=True, check_reconstruct=True)
            toks = map_single((sample_full_step, 'CMajor'))  # Throw in a random key
            mic(toks)
        else:
            tokenizer.s2c.omit_eos = True
            sample_full_step_ = ' '.join(sample_full_step.split()[:100])

            vocab_step = MusicVocabulary(pitch_kind='step')
            sample_txt2_cleaned = vocab_step.sanitize_rare_tokens(sample_full_step_)
            mic(sample_txt2_cleaned)
            toks = tokenizer.tokenize(sample_txt2_cleaned)
            mic(toks)

            ids = tokenizer(sample_txt2_cleaned).input_ids
            mic(ids)
    # check_trained_property()

    def check_trained_has_single_token():
        """
        Check each single token in vanilla vocab can be encoded into single token, 
        i.e. trained vocab has each single token, so that no `UNK` token needed
        """
        fnm = '22-10-24_WordPiece-Tokenizer_{dnm=all}_{vsz=32768, n=178825, pch=d, aug-key=T}'
        tokenizer = load_trained_tokenizer(fnm, pitch_kind='degree')
        vocab = tokenizer.vocab
        it = tqdm(vocab.tok2id.keys())
        for tok in it:
            encoded = tokenizer.encode(tok, entire_score=False)
            it.set_postfix(dict(tok=tok, encoded=encoded))

            cs = tokenizer.s2c.encode_single(tok)
            d_log = dict(token=tok, ids=encoded, chars=cs)
            print(pl.i(d_log))
            assert len(encoded) == 1
    # check_trained_has_single_token()

    def check_broken_if_has_rare_token():
        pk = 'step'
        vocab = MusicVocabulary(pitch_kind=pk, is_wordpiece=True)
        dnms = [pop]
        # dnms = [pop, mst]
        # dnms = [pop, mst, lmd]
        songs: List[Dict] = dataset.load_songs(*dnms)
        sr = transform.SanitizeRare(vocab=vocab)

        song = None
        for s in tqdm(songs, desc='Finding song w/ rare token'):
            if sr(s) != s:
                # mic(sr(s), s)
                song = s
                break
        mic(song[:100])

        fnm = '22-10-24_WordPiece-Tokenizer_{dnm=POP&MST}_{vsz=8192, n=2185, pch=s}'
        tokenizer = WordPieceMusicTokenizer.from_file(fnm, vocab=vocab)

        # inputs = tokenizer(sr(song))
        inputs = tokenizer(song)  # Will break if no sanitization
        mic(type(inputs))
        ids = inputs.input_ids
        mic(ids[:10])
    # check_broken_if_has_rare_token()

    def check_trained_tokenize_all():
        import random

        # pch_kd = 'midi'
        # pch_kd = 'step'
        pch_kd = 'degree'
        aug_key = pch_kd == 'degree'
        mic('Check trained tokenizer', pch_kd, aug_key)

        fnm = '22-11-26_WordPiece-Tokenizer_{dnm=all}_{vsz=262144, n=178825, pch=d, aug-key=T}'
        tokenizer = WordPieceMusicTokenizer.from_file(fnm, pitch_kind=pch_kd)

        # check_recon = True  # encoding & decoding reconstructs original text
        check_recon = False

        # sample = False
        sample = 6
        mic(check_recon, sample)

        # dnms = [pop]
        # dnms = [pop, mst]
        dnms = [pop, mst, lmd]
        _songs: List[Dict] = dataset.load_songs(*dnms)
        if aug_key:
            out = dataset.iter_song_w_all_keys(_songs)
            n, songs = out.total, out.generator
            if sample:  # total will be wrong...
                songs = (s for s in songs if random.randint(0, sample-1) == 0)
        else:
            n, songs = len(_songs), (s['score'] for s in _songs)

        fn = _CheckTrainedSingle(tokenizer=tokenizer, augment_key=aug_key, check_reconstruct=check_recon)

        # if aug_key and sample:  # Too many tokens, otherwise gets OOM
        #     def gen():
        #         for i, s in enumerate(songs):
        #             if random.randint(0, sample - 1) == 0:
        #                 yield s
        #     songs = gen()

        # concurrent = True
        concurrent = False
        c = Counter()
        with_tqdm = dict(desc='Checking trained tokenizer', unit='song', total=n)
        mic('about to launch')
        if concurrent:
            for toks in conc_yield(fn, songs, with_tqdm=with_tqdm, mode='process', batch_size=128):
                c.update(toks)
            # mic(c)
        else:
            c = Counter()
            for song in tqdm(songs, **with_tqdm):
                c.update(fn(song))
            # mic(c)
        c_ = dict()
        # Make sure file write is in the order of most common
        for tok, n in tqdm(c.most_common(), desc='Ordering counts for output', total=len(c)):
            c_[tok] = n
        with open(os_join(u.tokenizer_path, f'{fnm} distribution check.json'), 'w') as f:
            json.dump(dict(sample=sample, count=c_), f, indent=4)
    check_trained_tokenize_all()

    def check_id2pch():
        tokenizer = MusicTokenizer()
        ids = tokenizer.encode(sample_full_midi)
        pchs = tokenizer.ids2pitches(ids)
        mic(len(ids), len(pchs))

        wp_tokenizer = load_trained_tokenizer()
        ids = wp_tokenizer.encode(sample_full_midi)
        wp_pchs = wp_tokenizer.ids2pitches(ids)
        mic(len(ids), len(wp_pchs))
        assert wp_pchs == pchs
    # check_id2pch()

    def check_save():  # TODO
        tokenizer = load_trained_tokenizer(pitch_kind='degree')
        tokenizer.save_pretrained(os_join(u.tokenizer_path, 'tokenizer'))
    # check_save()
