from musicnlp.util import *
from musicnlp.postprocess import ElmType
from musicnlp.model import LMTTokenizer


class MusicConverter:
    def __init__(self, prec: int = 5, tokenizer_kw: Dict = None):
        self.prec = prec
        if tokenizer_kw is None:
            tokenizer_kw = dict()
        self.tokenizer = LMTTokenizer(prec=prec, **tokenizer_kw)
        self.vocab: MusicVocabulary = self.tokenizer.vocab

    def str2notes(self, toks: Union[str, List[str]]) -> List[Tuple[ElmType, Tuple]]:
        """
        Convert token string or pre-tokenized tokens into a compact format of music element tuples

        Expects each music element to be in the correct format
        """
        if isinstance(toks, str):
            toks = self.tokenizer._tokenize(toks)
        it = iter(toks)

        tok = next(it, None)
        lst_out = []
        while tok is not None:
            typ = self.vocab.type(tok)
            if typ == VocabType.special:
                if tok == self.vocab.start_of_bar:
                    lst_out.append((ElmType.bar_start, ()))
                elif tok == self.vocab.end_of_song:
                    lst_out.append((ElmType.song_end, ()))
                elif tok == self.vocab.start_of_tuplet:
                    tok = next(it, None)
                    toks_tup = []
                    while tok != self.vocab.end_of_tuplet:
                        toks_tup.append(tok)
                        tok = next(it, None)
                    assert len(toks_tup) >= 2
                    toks_p, tok_d = toks_tup[:-1], toks_tup[-1]
                    assert all(self.vocab.type(tok) == VocabType.pitch for tok in toks_p)
                    assert self.vocab.type(tok_d) == VocabType.duration
                    lst_out.append((
                        ElmType.tuplets,
                        (tuple([self.vocab.compact(tok) for tok in toks_p]), self.vocab.compact(tok_d))
                    ))
            elif typ == VocabType.time_sig:
                lst_out.append((ElmType.time_sig, (self.vocab.compact(tok))))
            elif typ == VocabType.tempo:
                lst_out.append((ElmType.tempo, (self.vocab.compact(tok))))
            else:
                assert typ == VocabType.pitch
                tok_d = next(it, None)
                assert self.vocab.type(tok_d) == VocabType.duration
                lst_out.append((ElmType.note, (self.vocab.compact(tok), self.vocab.compact(tok_d))))
            tok = next(it, None)
        return lst_out


if __name__ == '__main__':
    from icecream import ic

    mc = MusicConverter()
    text = get_extracted_song_eg()

    ic(mc.str2notes(text))
