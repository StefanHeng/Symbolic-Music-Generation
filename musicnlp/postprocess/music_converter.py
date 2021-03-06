from typing import List, Dict, Union

import numpy as np
import music21 as m21

from stefutil import *
from musicnlp.util.music_lib import *
from musicnlp.vocab import ElmType, MusicElement, VocabType, MusicVocabulary, MusicTokenizer
from musicnlp.preprocess import KeyFinder


class MusicConverter:
    def __init__(self, prec: int = 5, tokenizer_kw: Dict = None):
        self.prec = prec
        self.tokenizer = MusicTokenizer(precision=prec, **(tokenizer_kw or dict()))
        self.vocab: MusicVocabulary = self.tokenizer.vocab

    def _bar2grouped_bar(self, bar: Measure) -> List[ExtNote]:
        """
        Group triplet notes in the extracted music21 bar as tuples
        """
        it = iter(bar)
        elm = next(it, None)
        lst = []
        while elm is not None:  # similar logic as in `MusicExtractor.expand_bar`
            if hasattr(elm, 'fullName') and tuplet_postfix in elm.fullName:
                tup_str, n_tup = fullname2tuplet_meta(elm.fullName)
                lst_tup = [elm]
                elm_ = next(it, None)
                while elm_ is not None:
                    if tup_str in elm_.fullName:
                        lst_tup.append(elm_)
                        elm_ = next(it, None)
                    else:
                        break
                # Deal with consecutive and same-n tuplet groups in a row
                assert len(lst_tup) % n_tup == 0, \
                    f'Invalid Tuplet note count: {logi(tup_str)} with {logi(lst_tup)} should have ' \
                    f'multiples of {logi(n_tup)} notes'
                lst.extend([tuple(lst_tup_) for lst_tup_ in group_n(lst_tup, n_tup)])
                elm = elm_
                continue
            elif isinstance(elm, (Note, Rest)):
                lst.append(elm)
            elif isinstance(elm, Chord):  # TODO
                raise NotImplementedError('Chord not supported yet')
            elif not isinstance(elm, (TimeSignature, MetronomeMark)):  # which will be skipped
                raise ValueError(f'Unexpected element type: {logi(elm)} with type {logi(type(elm))}')
            elm = next(it, None)
        return lst

    def mxl2str(
            self, song: Union[str, Score], join: bool = True, n_bar: int = None, insert_key: Union[bool, str] = False
    ) -> Union[str, List[str]]:
        """
        Convert a MusicExtractor output song into the music token representation

        :param song: a music 21 Score or path to an MXL file
            Should be MusicExtractor output
        :param join: If true, individual tokens are jointed
        :param n_bar: If given, only return the decoded first n bars, star of bar is appended
                Intended for conditional generation
            Otherwise, the entire song is converted and end of song token is appended
        :param insert_key: A key is inserted accordingly, intended for generation
        """
        if isinstance(song, str):
            song = m21.converter.parse(song)
        song: Score
        parts = list(song.parts)
        warn = f'Check if the score is {logi("MusicExtractor")} output'
        # TODO: since melody only for now
        assert len(parts) == 1, f'Invalid #Parts: Expect only 1 part from the extracted score - {warn}'
        part = parts[0]
        bars = list(part[Measure])
        bar_nums = np.array([bar.number for bar in bars])
        assert np.array_equal(bar_nums, np.arange(0, len(bars))), \
            f'Invalid Bar numbers: Bar numbers should be consecutive integers starting from 0 - {warn}'
        bar0 = bars[0]
        time_sigs, tempos = bar0[TimeSignature], bar0[MetronomeMark]
        assert len(time_sigs) == 1, f'Invalid #Time Signatures: Expect only 1 time signature - {warn}'
        assert len(tempos) == 1, f'Invalid #Tempo: Expect only 1 tempo - {warn}'
        time_sig, tempo = time_sigs[0], tempos[0]
        toks = [self.vocab(time_sig), self.vocab(tempo)]
        if insert_key:
            key = insert_key if isinstance(insert_key, str) else pt_sample(KeyFinder(song).find_key(return_type='dict'))
            toks.append(self.vocab(key))

        for_gen = n_bar is not None
        if for_gen:
            assert n_bar > 0, f'Invalid {logi("n_bar")}: Expects positive integer'
            bars = bars[:min(n_bar, len(bars))]
        for bar in bars:
            assert all(not isinstance(e, m21.stream.Voice) for e in bar), f'Invalid Bar: Expect no voice - {warn}'
            toks.extend([[self.vocab.start_of_bar]] + [self.vocab(e) for e in self._bar2grouped_bar(bar)])
        # as `vocab` converts each music element to a list
        toks = sum(toks, start=[])
        toks += [self.vocab.start_of_bar if for_gen else self.vocab.end_of_song]
        return ' '.join(toks) if join else toks

    def str2notes(self, decoded: Union[str, List[str]]) -> List[MusicElement]:
        """
        Convert token string or pre-tokenized tokens into a compact format of music element tuples

        Expects each music element to be in the correct format
        """
        if isinstance(decoded, str):
            decoded = self.tokenizer.tokenize(decoded)
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
            elif typ == VocabType.key:  # ignore
                pass
            else:
                assert typ == VocabType.pitch
                tok_d = next(it, None)
                assert tok_d is not None and self.vocab.type(tok_d) == VocabType.duration, \
                    f'Pitch token {logi(tok)} should be followed by a duration token but got {logi(tok_d)}'
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
        assert note.type in [ElmType.note, ElmType.tuplets], \
            f'Invalid note type: expect one of {logi([ElmType.note, ElmType.tuplets])}, got {logi(note.type)}'

        pitch, q_len = note.meta
        dur = m21.duration.Duration(quarterLength=q_len)
        if note.type == ElmType.note:
            if pitch == -1:  # rest, see MusicVocabulary.compact
                return [Rest(duration=dur)]
            else:
                return [Note(pitch=m21.pitch.Pitch(midi=pitch), duration=dur)]
        else:  # tuplet
            dur_ea = quarter_len2fraction(q_len) / len(pitch)
            return sum([MusicConverter.note_elm2m21(MusicElement(ElmType.note, (p, dur_ea))) for p in pitch], start=[])

    def str2score(
            self, decoded: Union[str, List[str]], mode: str = 'melody', omit_eos: bool = False,
            title: str = None
    ) -> Score:
        """
        :param decoded: A string of list of tokens to convert to a music21 score
        :param mode: On of [`melody`, `chord`]
        :param omit_eos: If true, eos token at the end is not required for conversion
            All occurrences of eos in the sequence are ignored
        :param title: Title of the music
        """
        lst = self.str2notes(decoded)
        e1, e2, lst = lst[0], lst[1], lst[2:]
        assert e1.type == ElmType.time_sig, 'First element must be time signature'
        assert e2.type == ElmType.tempo, 'Second element must be tempo'
        if lst[0].type == ElmType.key:
            lst = lst[1:]  # ignore
        if omit_eos:
            lst = [e for e in lst if e.type != ElmType.song_end]
        else:
            lst, e_l = lst[:-1], lst[-1]
            assert e_l.type == ElmType.song_end, 'Last element must be end of song'
        idxs_bar_start = [i for i, e in enumerate(lst) if e.type == ElmType.bar_start]
        lst = [lst[idx:idxs_bar_start[i+1]] for i, idx in enumerate(idxs_bar_start[:-1])] + \
              [lst[idxs_bar_start[-1]:]]
        assert all((len(bar) > 1) for bar in lst), 'Bar should contain at least one note'
        lst = [sum([MusicConverter.note_elm2m21(n) for n in notes[1:]], start=[]) for notes in lst]
        time_sig = f'{e1.meta[0]}/{e1.meta[1]}'
        return make_score(title=title, mode=mode, time_sig=time_sig, tempo=e2.meta, lst_note=lst)


if __name__ == '__main__':
    from icecream import ic

    import musicnlp.util.music as music_util

    mc = MusicConverter()

    sample_text = 'TimeSig_4/4 Tempo_58 <bar> p_7/5 d_1/4 p_10/4 d_1/4 p_3/5 d_1/4 p_7/5 d_1/4 p_5/5 d_1/4 p_10/4 d_1/4 p_2/5 d_1/4 ' \
              'p_5/5 d_1/4 p_3/5 d_1/4 p_7/4 d_1/4 p_12/4 d_1/4 p_3/5 d_1/4 p_2/5 d_1/4 p_7/4 d_1/4 p_10/4 d_1/4 p_2/5 d_1/4 <bar> ' \
              'p_12/4 d_1/4 p_3/4 d_1/4 p_8/4 d_1/4 p_12/4 d_1/4 p_10/4 d_1/4 p_3/4 d_1/4 p_7/4 d_1/4 p_10/4 d_1/4 p_12/4 d_1/4 ' \
              'p_5/4 d_1/4 p_12/4 d_1/4 p_3/5 d_1/4 p_2/5 d_1/4 p_5/4 d_1/4 p_10/4 d_1/4 p_5/4 d_1/4 <bar> p_8/5 d_1 p_7/5 d_1 ' \
              'p_5/5 d_1 p_3/5 d_1 <bar> p_1/5 d_3/4 p_2/5 d_1/4 p_3/5 d_1/4 p_2/5 d_1/4 p_12/4 d_1/4 p_10/4 d_1/4 p_10/4 d_3/4 ' \
              'p_12/4 d_1/4 p_3/5 d_1 <bar> p_10/4 d_3/4 p_3/5 d_1/4 p_5/5 d_1/4 p_3/5 d_1/4 p_1/5 d_1/4 p_12/4 d_1/4 p_3/5 d_1/4 ' \
              'p_10/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/4 p_7/4 d_1/4 p_10/4 d_1/4 p_7/4 d_1/4 <bar> p_5/4 d_3/4 p_12/4 ' \
              'd_1/4 p_12/4 d_1/4 p_8/4 d_1/4 p_12/4 d_1/4 p_8/4 d_1/4 p_7/4 d_1/4 p_5/4 d_1/4 p_10/4 d_1/4 p_7/4 d_1/4 p_10/4 ' \
              'd_1/4 p_7/4 d_1/4 p_10/4 d_1/4 p_7/4 d_1/4 <bar> p_5/5 d_1 p_8/5 d_1 p_7/5 d_1 p_3/5 d_1 <bar> p_1/5 d_3/4 p_3/5 ' \
              'd_1/4 p_5/5 d_1/4 p_3/5 d_1/4 p_1/5 d_1/4 p_3/5 d_1/4 p_12/4 d_1/4 p_8/4 d_1/4 p_10/4 d_1/4 p_7/4 d_1/4 p_10/4 d_1/4 ' \
              'p_10/4 d_1/4 p_7/4 d_1/4 p_10/4 d_1/4 <bar> p_5/5 d_1 p_r d_1 p_r d_2 <bar> p_10/4 d_1 p_r d_1/2 <tup> p_12/5 p_8/5 ' \
              'p_3/5 p_10/5 p_8/5 d_2 </tup> <bar> p_7/5 d_1 p_10/4 d_1 p_8/5 d_2 <bar> p_12/4 d_1 p_r d_1/2 <tup> p_12/5 p_5/5 ' \
              'p_5/5 p_8/5 p_12/4 d_2 </tup> <bar> p_3/5 d_1 p_r d_1/2 <tup> p_r p_12/5 p_5/5 d_1/2 </tup> p_5/5 d_1/4 p_3/5 d_1/4 ' \
              'p_1/5 d_1/4 p_3/5 d_1/4 p_12/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/4 <bar> p_10/4 d_1/4 p_10/4 d_1/4 p_10/4 ' \
              'd_1/4 p_10/4 d_1/4 p_7/4 d_1/4 p_3/5 d_1/4 p_3/5 d_1/4 p_5/5 d_1/4 p_5/5 d_1/2 p_5/5 d_1/4 p_3/5 d_1/4 p_1/5 d_1/4 ' \
              'p_12/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/4 <bar> p_2/5 d_1/2 p_3/5 d_1/4 p_3/5 d_1/4 p_3/5 d_1/4 p_5/5 d_1/4 p_5/5 d_1/4 ' \
              'p_3/5 d_1/4 p_1/5 d_1/4 p_3/5 d_1/4 p_12/4 d_1/4 p_10/4 d_1/4 p_8/5 d_1/2 p_7/5 d_1/2 <bar> p_5/5 d_1 p_r d_1/2 p_r ' \
              'd_1/4 p_5/4 d_1/4 p_7/4 d_3/4 p_8/4 d_1/4 p_12/4 d_1/4 p_10/4 d_1/4 p_3/5 d_1/4 p_2/5 d_1/4 <bar> p_5/5 d_1 p_r ' \
              'd_1/2 p_r d_1/4 p_5/4 d_1/4 p_7/4 d_1/4 p_8/4 d_1/4 p_10/4 d_1/2 p_10/4 d_1/4 p_12/4 d_1/4 p_1/5 d_1/4 p_3/5 d_1/4 ' \
              '<bar> p_5/5 d_1 p_10/5 d_1 p_2/5 d_1/2 p_3/5 d_1/4 p_3/5 d_1/4 p_1/5 d_1/4 p_12/4 d_1/4 p_3/5 d_1/4 p_2/5 d_1/4 ' \
              '<bar> p_3/5 d_1 p_r d_1/8 p_r d_1/8 p_7/4 d_1/4 p_8/4 d_1/4 p_11/4 d_1/4 p_3/5 d_1/4 p_5/5 d_1 p_8/5 d_1 <bar> p_7/5 ' \
              'd_1 p_1/5 d_3/4 p_7/5 d_1/4 p_8/5 d_1 p_10/5 d_1 <bar> p_8/5 d_4 <bar> p_8/4 d_3/4 p_9/5 d_1/4 p_10/5 d_3/4 p_3/5 ' \
              'd_1/4 p_5/5 d_1 p_6/5 d_1/2 p_5/5 d_1/2 <bar> p_8/5 d_1 p_8/5 d_1/2 p_6/5 d_1/4 p_5/5 d_1/4 p_3/5 d_2 <bar> p_3/5 ' \
              'd_3/4 p_1/6 d_1/4 p_1/6 d_1 p_10/5 d_1/4 p_5/5 d_3/4 p_12/4 d_1 <bar> p_7/5 d_1 p_5/5 d_1/4 p_3/5 d_1/4 p_5/5 d_1/4 ' \
              'p_3/5 d_1/4 p_1/5 d_1/4 p_12/4 d_1/4 p_1/5 d_3/4 p_3/5 d_1/4 <bar> p_5/5 d_1 p_6/5 d_1 p_8/5 d_1 p_3/4 d_1/2 p_10/4 ' \
              'd_1/4 p_12/4 d_1/4 <bar> p_5/5 d_1 p_5/5 d_1 p_8/5 d_1 p_12/5 d_1 <bar> p_10/5 d_4 <bar> p_8/5 d_1 p_12/5 d_1 p_1/6 ' \
              'd_1/4 p_12/4 d_1/4 p_12/4 d_1/2 p_6/5 d_1 <bar> p_1/3 d_1/8 p_12/5 d_1/8 p_10/5 d_3/4 p_12/5 d_1 p_1/6 d_3/4 p_8/5 ' \
              'd_1/4 p_12/4 d_1/4 p_8/5 d_1/4 p_7/5 d_1/4 p_5/5 d_1/4 <bar> p_1/5 d_1/2 p_2/5 d_1/4 p_3/5 d_1/4 p_1/5 d_1/4 p_3/5 ' \
              'd_1/4 p_12/4 d_1/4 p_10/4 d_1/4 p_10/5 d_1 p_10/5 d_1 <bar> p_12/4 d_1/2 p_10/4 d_1/4 p_10/4 d_1/4 p_7/4 d_1/4 p_3/5 ' \
              'd_1/4 p_3/5 d_1/4 p_10/4 d_1/4 p_12/4 d_1/4 p_10/4 d_1/4 p_8/5 d_3/2 <bar> p_3/5 d_1 p_3/5 d_1/4 p_7/5 d_1/4 p_7/5 ' \
              'd_1/4 p_3/5 d_1/4 p_10/4 d_1/4 p_7/5 d_1/4 p_3/5 d_1/4 p_10/4 d_1/4 p_7/5 d_1 <bar> p_10/5 d_1 p_10/5 d_1 p_3/6 d_1 ' \
              'p_1/6 d_1 <bar> p_r d_3/8 p_10/5 d_1/8 p_10/5 d_1/2 p_12/5 d_1/2 p_1/6 d_1/2 p_12/5 d_1/2 p_1/6 d_1/2 p_12/5 d_1/2 ' \
              'p_1/6 d_1/2 <bar> p_8/5 d_1/4 p_10/5 d_1/4 p_8/5 d_1/4 p_7/5 d_1/4 p_5/5 d_1/4 p_1/5 d_1/4 p_3/5 d_1/4 p_7/5 d_1/4 ' \
              'p_7/5 d_1/4 p_3/5 d_1/4 p_1/5 d_1/4 p_7/4 d_1/4 p_10/4 d_1/4 p_5/5 d_1/4 p_3/5 d_1/4 p_7/5 d_1/4 <bar> p_7/5 d_1/4 ' \
              'p_3/5 d_1/4 p_2/5 d_1/4 p_3/5 d_1/4 p_5/5 d_1/4 p_3/5 d_1/4 p_5/5 d_1/8 p_3/5 d_1/8 p_2/5 d_1/4 p_10/5 d_1/8 p_10/5 ' \
              'd_1/8 p_7/5 d_1/8 p_3/5 d_1/8 p_2/5 d_1/8 p_12/4 d_1/8 p_9/4 d_1/4 p_10/4 d_1/8 p_10/4 d_1/8 p_7/4 d_1/2 <bar> p_8/4 ' \
              'd_3/4 p_7/4 d_1/4 p_8/4 d_1/4 p_7/4 d_1/4 p_12/3 d_1/4 p_8/3 d_1/4 p_8/3 d_1/4 p_8/3 d_1/4 p_3/3 d_1/4 <tup> p_3/3 ' \
              'p_3/3 p_12/2 d_1/2 </tup> <tup> p_10/2 p_3/3 p_5/3 d_1/2 </tup> <tup> p_12/2 p_10/2 p_11/2 d_1/2 </tup> <bar> p_5/4 ' \
              'd_3/4 p_2/4 d_1/4 p_3/4 d_1/4 p_3/4 d_1/4 p_7/3 d_1/4 p_12/3 d_1/4 p_12/2 d_1/4 p_8/3 d_1/4 p_8/3 d_1/4 p_7/3 d_1/4 ' \
              '<tup> p_12/3 p_5/3 p_5/3 d_1/2 </tup> <tup> p_3/4 p_8/3 p_5/3 d_1/2 </tup> <bar> <tup> p_8/3 p_5/3 p_7/3 d_1/2 ' \
              '</tup> <tup> p_5/3 p_8/3 p_10/3 d_1/2 </tup> <tup> p_8/2 p_1/4 p_2/4 d_1/2 </tup> <tup> p_5/2 p_9/3 p_12/3 d_1/2 ' \
              '</tup> <tup> p_5/2 p_5/3 p_5/3 d_1/2 </tup> <tup> p_12/2 p_10/2 p_10/3 d_1/2 </tup> p_3/4 d_1/2 <bar> p_3/4 d_1 ' \
              'p_3/4 d_1/2 p_12/3 d_1/2 p_12/3 d_1/2 p_8/3 d_1/2 p_2/4 d_1/2 p_2/4 d_1/2 <bar> p_3/4 d_1 p_8/3 d_1 p_7/3 d_1/2 ' \
              'p_8/3 d_1/4 p_7/3 d_1/4 p_8/3 d_1/2 p_10/3 d_1/4 <bar> p_10/2 d_3/4 p_10/5 d_1/4 p_10/5 d_3 <bar> p_1/3 d_1/2 p_9/2 ' \
              'd_1/4 p_12/2 d_1/4 p_10/3 d_1/2 <tup> p_1/3 p_1/3 p_1/4 d_1/2 </tup> p_r d_1/8 p_1/4 d_1/8 <tup> p_1/3 p_8/3 p_r ' \
              'd_1/2 </tup> p_10/1 d_1/8 p_7/3 d_1/4 p_5/4 d_1/2 p_8/4 d_1/2 <bar> p_5/5 d_1/2 <tup> p_3/5 p_2/5 p_8/4 d_1/2 </tup> ' \
              'p_2/5 d_1/4 p_2/5 d_1/4 p_11/4 d_1/4 p_8/4 d_1/4 p_7/4 d_1/2 <tup> p_4/4 p_7/4 p_3/3 d_1/2 </tup> p_5/5 d_1/4 p_3/5 ' \
              'd_1/4 p_2/5 d_1/4 p_12/4 d_1/4 p_10/4 d_1/4 p_12/5 d_1 <bar> p_3/4 d_2 p_3/5 d_2 <bar> p_8/5 d_2 p_7/5 d_2 <bar> ' \
              'p_7/5 d_2 p_10/5 d_1 p_5/5 d_1 <bar> p_11/5 d_1 p_10/5 d_1 p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1 <bar> p_10/5 d_1/2 ' \
              'p_7/5 d_1/2 p_6/5 d_1/2 p_7/5 d_1/2 p_9/5 d_3/4 p_5/2 d_1/4 <tup> p_6/5 p_5/5 p_9/5 d_1 </tup> <bar> p_8/5 d_1 p_9/5 ' \
              'd_1/4 p_5/5 d_1/4 p_9/5 d_1/8 p_r d_1/8 p_7/5 d_1/4 p_6/5 d_1/2 p_7/5 d_1/2 p_6/5 d_1/2 p_7/5 d_1/2 p_6/5 d_1/2 ' \
              '<bar> p_5/5 d_1/4 p_9/5 d_3/4 p_8/5 d_1/2 p_9/5 d_1/2 p_7/5 d_1/2 p_6/5 d_1/2 p_7/5 d_1/2 p_6/5 d_1/2 <bar> p_4/5 ' \
              'd_5/4 p_8/5 d_1/4 p_11/5 d_1 p_11/5 d_1'

    def check_encode():
        # text = get_extracted_song_eg(k=2)  # this one has tuplets
        # text = music_util.get_extracted_song_eg(k='????????????')  # this one has tuplets
        # ic(text)
        # toks = mc.str2notes(text)
        # ic(toks)
        # scr = mc.str2score(text)
        scr = mc.str2score(sample_text, omit_eos=True)
        ic(scr)
        # scr.show()
    check_encode()

    def check_decode():
        # fnm = 'Shape of You'
        fnm = 'Merry Go Round'
        path = music_util.get_my_example_songs(k=fnm, extracted=True)
        ic(path)
        # ic(mc.mxl2str(path))
        ic(mc.mxl2str(path, n_bar=4))
    # check_decode()
