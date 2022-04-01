from musicnlp.util import *
from musicnlp.util.music_lib import *
from musicnlp.postprocess import ElmType, MusicElement
from musicnlp.model import MusicTokenizer


class MusicConverter:
    def __init__(self, prec: int = 5, tokenizer_kw: Dict = None):
        self.prec = prec
        if tokenizer_kw is None:
            tokenizer_kw = dict()
        self.tokenizer = MusicTokenizer(prec=prec, **tokenizer_kw)
        self.vocab: MusicVocabulary = self.tokenizer.vocab

    def str2notes(self, decoded: Union[str, List[str]]) -> List[MusicElement]:
        """
        Convert token string or pre-tokenized tokens into a compact format of music element tuples

        Expects each music element to be in the correct format
        """
        if isinstance(decoded, str):
            decoded = self.tokenizer._tokenize(decoded)
        it = iter(decoded)

        tok = next(it, None)
        lst_out = []
        while tok is not None:
            typ = self.vocab.type(tok)
            if typ == VocabType.special:
                if tok == self.vocab.start_of_bar:
                    lst_out.append(MusicElement(type=ElmType.bar_start, meta=None))
                elif tok == self.vocab.end_of_song:
                    lst_out.append(MusicElement(type=ElmType.song_end, meta=None))
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
                    lst_out.append(MusicElement(
                        type=ElmType.tuplets,
                        meta=(tuple([self.vocab.compact(tok) for tok in toks_p]), self.vocab.compact(tok_d))
                    ))
            elif typ == VocabType.time_sig:
                lst_out.append(MusicElement(type=ElmType.time_sig, meta=(self.vocab.compact(tok))))
            elif typ == VocabType.tempo:
                lst_out.append(MusicElement(type=ElmType.tempo, meta=(self.vocab.compact(tok))))
            else:
                assert typ == VocabType.pitch
                tok_d = next(it, None)
                assert self.vocab.type(tok_d) == VocabType.duration
                lst_out.append(
                    MusicElement(type=ElmType.note, meta=(self.vocab.compact(tok), self.vocab.compact(tok_d)))
                )
            tok = next(it, None)
        return lst_out

    @staticmethod
    def note_elm2m21(note: MusicElement) -> List[SNote]:
        """
        Convert a music element tuple into a music21 note or tuplet of notes
        """
        # ic(note)
        assert note.type in [ElmType.note, ElmType.tuplets], \
            f'Invalid note type: expect one of {logi([ElmType.note, ElmType.tuplets])}, got {logi(note.type)}'

        pitch, q_len = note.meta
        dur = m21.duration.Duration(quarterLength=q_len)
        if note.type == ElmType.note:
            if pitch == -1:  # rest, see MusicVocabulary.compact
                return [m21.note.Rest(duration=dur)]
            else:
                return [Note(pitch=m21.pitch.Pitch(midi=pitch), duration=dur)]
        else:  # tuplet
            dur_ea = quarter_len2fraction(q_len) / len(pitch)
            return sum([MusicConverter.note_elm2m21(MusicElement(ElmType.note, (p, dur_ea))) for p in pitch], start=[])

    def str2score(self, decoded: Union[str, List[str]], mode: str = 'melody') -> Score:
        lst = self.str2notes(decoded)
        e1, e2, lst, e_l = lst[0], lst[1], lst[2:-1], lst[-1]
        assert e1.type == ElmType.time_sig, 'First element must be time signature'
        assert e2.type == ElmType.tempo, 'Second element must be tempo'
        assert e_l.type == ElmType.song_end, 'Last element must be end of song'

        idxs_bar_start = [i for i, e in enumerate(lst) if e.type == ElmType.bar_start]
        lst = [lst[idx:idxs_bar_start[i+1]] for i, idx in enumerate(idxs_bar_start[:-1])] + \
              [lst[idxs_bar_start[-1]:]]
        assert all((len(bar) > 1) for bar in lst), 'Bar should contain at least one note'
        # ic(lst)
        lst = [sum([MusicConverter.note_elm2m21(n) for n in notes[1:]], start=[]) for notes in lst]
        return make_score(
            title='Generated', mode=mode, time_sig=f'{e1.meta[0]}/{e1.meta[1]}', tempo=e2.meta, lst_note=lst
        )


if __name__ == '__main__':
    from icecream import ic

    mc = MusicConverter()
    # text = get_extracted_song_eg(k=2)  # this one has tuplets
    text = get_extracted_song_eg(k='平凡之路')  # this one has tuplets
    toks = mc.str2notes(text)
    # ic(text, toks)
    scr = mc.str2score(text)
    ic(scr)
    scr.show()
