from typing import List, Dict, Union
from dataclasses import dataclass

import music21 as m21

from stefutil import *
from musicnlp.util.music_lib import *
from musicnlp.vocab import Song, ElmType, Channel, MusicElement, VocabType, MusicVocabulary
from musicnlp.preprocess import KeyFinder


__all__ = [
    'MusicConverter',
    'MusicElm', 'SongSplitOutput'
]


@dataclass
class PartExtractOutput:
    time_sig: TsTup = None
    tempo: int = None
    key: str = None
    toks: List[List[str]] = None  # List of tokens for each bar


@dataclass
class ElmParseOutput:
    elms: List[MusicElement] = None
    time_sig: MusicElement = None
    tempo: MusicElement = None
    key: MusicElement = None
    elms_by_bar: List[List[MusicElement]] = None


MusicElm = List[str]


@dataclass
class SongSplitOutput:
    elms: List[MusicElm] = None
    time_sig: str = None
    tempo: str = None
    key: str = None
    omit: str = None
    elms_by_bar: List[List[MusicElm]] = None
    end_of_song: str = None


@dataclass
class MusicVocabs:
    midi: MusicVocabulary = None
    step: MusicVocabulary = None
    degree: MusicVocabulary = None


class MusicConverter:
    error_prefix = 'MusicConvertor Song Input Format Check'

    def __init__(
            self, mode: str = 'full', precision: int = 5, vocab_midi: MusicVocabulary = None,
            vocab_step: MusicVocabulary = None, vocab_degree: MusicVocabulary = None, augment_key: bool = False
    ):
        ca(extract_mode=mode)
        self.mode = mode

        self.vocabs: MusicVocabs = MusicVocabs(
            midi=vocab_midi or MusicVocabulary(pitch_kind='midi'),
            step=vocab_step or MusicVocabulary(pitch_kind='step'),
            degree=vocab_degree or MusicVocabulary(pitch_kind='degree')
        )
        for v in [self.vocabs.midi, self.vocabs.step, self.vocabs.degree]:
            assert v.precision == precision
        self.augment_key = augment_key

        _vocab = self.vocabs.midi  # doesn't matter which one
        self._non_tup_spec = {
            _vocab.omitted_segment,
            _vocab.start_of_bar, _vocab.end_of_song, _vocab.start_of_melody, _vocab.start_of_bass
        }

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
                    f'Invalid Tuplet note count: {pl.i(tup_str)} with {pl.i(lst_tup)} should have ' \
                    f'multiples of {pl.i(n_tup)} notes'
                lst.extend([tuple(lst_tup_) for lst_tup_ in group_n(lst_tup, n_tup)])
                elm = elm_
                continue
            elif isinstance(elm, (Note, Rest)):
                lst.append(elm)
            elif isinstance(elm, Chord):  # TODO
                raise NotImplementedError('Chord not supported yet')
            elif not isinstance(elm, (TimeSignature, MetronomeMark, m21.clef.BassClef)):  # which will be skipped
                raise ValueError(f'Unexpected element type: {pl.i(elm)} with type {pl.i(type(elm))}')
            elm = next(it, None)
        return lst

    @staticmethod
    def _input_format_error(check: bool, msg: str):
        if not check:
            raise ValueError(f'{MusicConverter.error_prefix}: {msg}')

    def _part2toks(
            self, part: Part, insert_key: str = None, n_bar: int = None, song: Union[str, Score] = None,
            check_meta: bool = True
    ) -> PartExtractOutput:
        bars = list(part[Measure])
        MusicConverter._input_format_error(
            [bar.number for bar in bars] == list(range(len(bars))), f'Invalid #Bar numbers ')
        bar0 = bars[0]
        time_sigs, tempos = bar0[TimeSignature], bar0[MetronomeMark]
        if check_meta:
            MusicConverter._input_format_error(len(time_sigs) == 1, f'Expect only 1 time signature ')
            MusicConverter._input_format_error(len(tempos) == 1, f'Expect only 1 tempo ')
        time_sig, tempo = next(time_sigs, None), next(tempos, None)
        ts_tup = (time_sig.numerator, time_sig.denominator) if time_sig else None
        if tempo:
            tempo = tempo.number
            assert tempo.is_integer()
            tempo = int(tempo)
        else:
            tempo = None

        key = None
        if insert_key:
            key = insert_key if isinstance(insert_key, str) else pt_sample(KeyFinder(song)(return_type='dict'))

        if n_bar is not None:  # for conditional generation
            MusicConverter._input_format_error(n_bar > 0, f'{pl.i("n_bar")} should be positive integer')
            bars = bars[:min(n_bar, len(bars))]
        toks = []

        vocab = self.vocabs.step if self.augment_key else self.vocabs.midi
        for i, bar in enumerate(bars):
            MusicConverter._input_format_error(
                all(not isinstance(e, m21.stream.Voice) for e in bar), f'Expect no voice in bar#{pl.i(i)}')
            # as `vocab` converts each music element to a list
            toks.append(sum([vocab(e) for e in self._bar2grouped_bar(bar)], start=[]))
        return PartExtractOutput(time_sig=ts_tup, tempo=tempo, key=key, toks=toks)

    def mxl2str(
            self, song: Union[str, Score], join: bool = True, n_bar: int = None, insert_key: Union[bool, str] = False
    ) -> Union[str, List[str]]:
        """
        Convert a MusicExtractor output song into the music token representation

        :param song: a music 21 Score or path to an MXL file
            Should be MusicExtractor output
        :param join: If true, individual tokens are jointed
        :param n_bar: If given, only return the decoded first n bars, star of bar is appended at the end
                Intended for conditional generation
            Otherwise, the entire song is converted and end of song token is appended
        :param insert_key: A key is inserted accordingly, intended for generation
        """
        vocab = self.vocabs.step if self.augment_key else self.vocabs.midi

        if isinstance(song, str):
            song = m21.converter.parse(song)
        song: Score
        parts = list(song.parts)
        correct_n_part = (self.mode == 'melody' and len(parts) == 1) or (self.mode == 'full' and len(parts) == 2)
        MusicConverter._input_format_error(correct_n_part, f'Invalid #Part for f{pl.i(self.mode)} ')
        part_melody = next(p for p in parts if 'Melody' in p.partName)  # See `make_score`
        part_bass = None
        if self.mode == 'full':
            part_bass = next(p for p in parts if 'Bass' in p.partName)

        out_m = self._part2toks(part=part_melody, insert_key=insert_key, n_bar=n_bar, song=song)
        time_sig, tempo, key = out_m.time_sig, out_m.tempo, out_m.key
        out_b = None
        if self.mode == 'full':  # sanity check
            out_b = self._part2toks(part=part_bass, insert_key=insert_key, n_bar=n_bar, song=song, check_meta=False)
            assert not out_b.time_sig or time_sig == out_b.time_sig
            assert not out_b.tempo or tempo == out_b.tempo
            assert not out_b.key or key == out_b.key
        toks = [vocab(time_sig)[0], vocab(tempo)[0]]
        if insert_key:
            toks.append(vocab(key)[0])

        if self.mode == 'melody':
            for ts in out_m.toks:
                toks.append(vocab.start_of_bar)
                toks.extend(ts)
        else:  # 'full'
            for ts_m, ts_b in zip(out_m.toks, out_b.toks):
                toks.extend([vocab.start_of_bar, vocab.start_of_melody])
                toks.extend(ts_m)
                toks.append(vocab.start_of_bass)
                toks.extend(ts_b)
        for_gen = n_bar is not None
        toks += [vocab.start_of_bar if for_gen else vocab.end_of_song]
        return ' '.join(toks) if join else toks

    def str2tok_elms(self, text: Song) -> SongSplitOutput:
        """
        Like `str2music_elms`, but split into token groups only, without the conversion to `MusicElement`
        """
        toks = text if isinstance(text, list) else text.split()
        elms: List[MusicElm] = []
        it = iter(toks)
        tok = next(it, None)

        vocab = self.vocabs.midi  # doesn't matter which one
        while tok is not None:
            typ = vocab.type(tok)
            if typ == VocabType.special:
                if tok in self._non_tup_spec:
                    elms.append([tok])
                else:
                    assert tok == vocab.start_of_tuplet  # sanity check
                    tok = next(it, None)
                    toks_tup = []
                    while tok != vocab.end_of_tuplet:
                        toks_tup.append(tok)
                        tok = next(it, None)  # in the end, consumes `end_of_tuplet` token
                    toks_p, tok_d = toks_tup[:-1], toks_tup[-1]
                    assert len(toks_tup) >= 3  # sanity check
                    assert all(vocab.type(t) == VocabType.pitch for t in toks_p)
                    assert vocab.type(tok_d) == VocabType.duration
                    elms.append([vocab.start_of_tuplet, *toks_p, tok_d, vocab.end_of_tuplet])
            elif typ in [VocabType.time_sig, VocabType.tempo, VocabType.key]:
                elms.append([tok])
            else:
                assert typ == VocabType.pitch
                tok_d = next(it, None)
                assert vocab.type(tok_d) == VocabType.duration
                elms.append([tok, tok_d])
            tok = next(it, None)

        ts, tp, key, omit, elms = elms[0], elms[1], None, None, elms[2:]
        assert vocab.type(ts[0]) == VocabType.time_sig
        assert vocab.type(tp[0]) == VocabType.tempo
        ts, tp = ts[0], tp[0]
        if vocab.type(elms[0][0]) == VocabType.key:
            key = elms[0][0]
            elms = elms[1:]
        if elms[0][0] == vocab.omitted_segment:
            omit = elms[0][0]
            elms = elms[1:]

        idxs_bar = [i for i, es in enumerate(elms) if es == [vocab.start_of_bar]]
        elms_by_bar = [elms[idx:idxs_bar[i+1]] for i, idx in enumerate(idxs_bar[:-1])] + [elms[idxs_bar[-1]:]]
        elms_by_bar = [es[1:] for es in elms_by_bar]  # skip the bar start token

        eos = None
        if elms_by_bar[-1][-1] == [vocab.end_of_song]:
            elms_by_bar[-1] = elms_by_bar[-1][:-1]
            eos = vocab.end_of_song
        return SongSplitOutput(
            elms=elms, time_sig=ts, tempo=tp, key=key, omit=omit, elms_by_bar=elms_by_bar, end_of_song=eos
        )

    def visualize_str(self, score: Union[str, List[str], List[List[str]]]):
        """
        Visualize a complete song in tokens, one bar per row
        """
        groups: List[List[str]]
        if isinstance(score, (list, tuple)) and isinstance(score[0], (list, tuple)):
            groups = score
        else:
            out = self.str2tok_elms(score)
            groups = [[g for g in [out.time_sig, out.tempo, out.key, out.omit] if g is not None]]
            groups += [sum(elms, start=[]) for elms in out.elms_by_bar]
        n_pad = len(str(len(groups)))

        def map_prefix(i_row):
            return pl.s(f'{i_row:>{n_pad}}:', c='y')
        vocab = self.vocabs.degree if self.augment_key else self.vocabs.midi
        return '\n'.join(f'{map_prefix(i)} {vocab.colorize_tokens(toks)}' for i, toks in enumerate(groups))

    def str2music_elms(
            self, text: Song, group: bool = True, omit_eos: bool = False, strict: bool = True
    ) -> ElmParseOutput:
        """
        Convert token string or pre-tokenized tokens into a compact format of music element tuples

        Expects each music element to be in the correct format
        """
        vocab = self.vocabs.degree if self.augment_key else self.vocabs.midi

        def comp(x):  # syntactic sugar
            return vocab.tok2meta(x, strict=strict)

        text = text if isinstance(text, list) else text.split()
        it = iter(text)

        tok = next(it, None)
        lst_out = []
        while tok is not None:
            # mic(tok)
            typ = vocab.type(tok)
            if typ == VocabType.special:
                if tok == vocab.start_of_bar:
                    lst_out.append(MusicElement(type=ElmType.bar_start, meta=None))
                elif tok == vocab.end_of_song:
                    lst_out.append(MusicElement(type=ElmType.song_end, meta=None))
                elif tok == vocab.start_of_tuplet:
                    tok = next(it, None)
                    toks_tup = []
                    while tok != vocab.end_of_tuplet:
                        toks_tup.append(tok)
                        tok = next(it, None)
                    assert len(toks_tup) >= 2
                    toks_p, tok_d = toks_tup[:-1], toks_tup[-1]
                    assert all(vocab.type(tok) == VocabType.pitch for tok in toks_p)
                    assert vocab.type(tok_d) == VocabType.duration
                    lst_out.append(MusicElement(
                        type=ElmType.tuplets, meta=(tuple([comp(tok) for tok in toks_p]), comp(tok_d))
                    ))
                elif tok == vocab.start_of_melody:
                    assert self.mode == 'full'
                    lst_out.append(MusicElement(type=ElmType.melody, meta=None))
                else:
                    assert tok == vocab.start_of_bass
                    assert self.mode == 'full'
                    lst_out.append(MusicElement(type=ElmType.bass, meta=None))
            elif typ == VocabType.time_sig:
                lst_out.append(MusicElement(type=ElmType.time_sig, meta=(comp(tok))))
            elif typ == VocabType.tempo:
                lst_out.append(MusicElement(type=ElmType.tempo, meta=(comp(tok))))
            elif typ == VocabType.key:
                lst_out.append(MusicElement(type=ElmType.key, meta=(comp(tok))))
            else:
                assert typ == VocabType.pitch
                tok_d = next(it, None)
                assert tok_d is not None and vocab.type(tok_d) == VocabType.duration, \
                    f'Pitch token {pl.i(tok)} should be followed by a duration token but got {pl.i(tok_d)}'
                lst_out.append(MusicElement(type=ElmType.note, meta=(comp(tok), comp(tok_d))))
            tok = next(it, None)

        ts, tp, key, bar_lst = None, None, None, None
        if group:
            ts, tp, lst = lst_out[0], lst_out[1], lst_out[2:]
            assert ts.type == ElmType.time_sig, 'First element must be time signature'
            assert tp.type == ElmType.tempo, 'Second element must be tempo'
            if lst[0].type == ElmType.key:  # ignore
                key, lst = lst[0], lst[1:]
            if omit_eos:
                lst = [e for e in lst if e.type != ElmType.song_end]
            else:
                lst, e_l = lst[:-1], lst[-1]
                assert e_l.type == ElmType.song_end, 'Last element must be end of song'
            idxs_bar_start = [i for i, e in enumerate(lst) if e.type == ElmType.bar_start]
            bar_lst = [lst[idx:idxs_bar_start[i + 1]] for i, idx in enumerate(idxs_bar_start[:-1])] + \
                      [lst[idxs_bar_start[-1]:]]
            bar_lst = [notes[1:] for notes in bar_lst]  # by construction, the 1st element is `bar_start`
            assert all((len(bar) > 0) for bar in bar_lst), 'Bar should contain at least one note'
        return ElmParseOutput(elms=lst_out, time_sig=ts, tempo=tp, key=key, elms_by_bar=bar_lst)

    @staticmethod
    def mus_elm2m21_note(note: MusicElement) -> List[SNote]:
        """
        Convert a music element tuple into a music21 note or tuplet of notes
        """
        assert note.type in [ElmType.note, ElmType.tuplets], \
            f'Invalid note type: expect one of {pl.i([ElmType.note, ElmType.tuplets])}, got {pl.i(note.type)}'

        pitch, q_len = note.meta
        if isinstance(pitch, tuple):  # 2nd element is step/scale degree, ignore; TODO: add to the rendered MXL?
            pitch = pitch[0]
        dur = m21.duration.Duration(quarterLength=q_len)
        if note.type == ElmType.note:
            if pitch == MusicVocabulary.rest_pitch_code:
                return [Rest(duration=dur)]
            else:
                return [Note(pitch=m21.pitch.Pitch(midi=pitch), duration=dur)]
        else:  # tuplet
            dur_ea = quarter_len2fraction(q_len) / len(pitch)
            return sum(
                [MusicConverter.mus_elm2m21_note(MusicElement(ElmType.note, (p, dur_ea))) for p in pitch], start=[]
            )

    @staticmethod
    def bar2notes(notes: List[MusicElement]) -> List[SNote]:
        return sum([MusicConverter.mus_elm2m21_note(n) for n in notes], start=[])

    @staticmethod
    def split_notes(notes: List[MusicElement]) -> Dict[str, List[MusicElement]]:
        """
        Split the notes in a bar into two lists for Melody & Bass
        """
        lst_melody, lst_bass = [], []
        it = iter(notes)
        c = Channel.melody if next(it).type == ElmType.melody else Channel.bass  # 1st note have to specify this

        n = next(it, None)
        while n is not None:
            if n.type == ElmType.melody:
                c = Channel.melody
            elif n.type == ElmType.bass:
                c = Channel.bass
            else:
                (lst_melody if c == Channel.melody else lst_bass).append(n)
            n = next(it, None)
        return dict(melody=lst_melody, bass=lst_bass)

    def str2score(
            self, decoded: Union[str, List[str]], omit_eos: bool = False,
            title: str = None, check_duration_match: bool = True
    ) -> Score:
        """
        :param decoded: A string of list of tokens to convert to a music21 score
        :param omit_eos: If true, eos token at the end is not required for conversion
            All occurrences of eos in the sequence are ignored
        :param title: Title of the music
        :param check_duration_match: See `make_score`
        """
        lst = self.str2music_elms(decoded, group=True, omit_eos=omit_eos)
        ts, tp, key, lst = lst.time_sig, lst.tempo, lst.key, lst.elms_by_bar

        if self.mode == 'melody':
            d_notes = dict(melody=[MusicConverter.bar2notes(notes) for notes in lst])
        else:  # `full`
            d_notes = dict(melody=[], bass=[])
            for notes in lst:
                d = MusicConverter.split_notes(notes)
                d_notes['melody'].append(MusicConverter.bar2notes(d['melody']))
                d_notes['bass'].append(MusicConverter.bar2notes(d['bass']))
        time_sig = f'{ts.meta[0]}/{ts.meta[1]}'
        return make_score(
            title=title, mode=self.mode, time_sig=time_sig, tempo=tp.meta, d_notes=d_notes,
            check_duration_match=check_duration_match
        )


if __name__ == '__main__':
    import musicnlp.util.music as music_util
    from musicnlp._sample_score import sample_full_midi, gen_broken

    md = 'full'

    def check_map_elm():
        mc = MusicConverter(mode=md)
        text = music_util.get_extracted_song_eg(k='平凡之路')
        mic(text)
        toks = mc.str2music_elms(text)
        mic(toks)
        scr = mc.str2score(text)
        scr.show()
    # check_map_elm()

    def check_encode():
        mc = MusicConverter(mode=md)
        scr = mc.str2score(sample_full_midi, omit_eos=True, title='Test')
        mic(scr)
        scr.show()
    # check_encode()

    def check_decode():
        mc = MusicConverter(mode=md)
        # fnm = 'Shape of You'
        fnm = 'Merry Go Round'
        path = music_util.get_my_example_songs(k=fnm, extracted=True)
        mic(path)
        # mic(mc.mxl2str(path))
        mic(mc.mxl2str(path, n_bar=4))
    # check_decode()

    def check_encode_decode():
        mc = MusicConverter(mode=md)
        fnm = '平凡之路'
        path = music_util.get_my_example_songs(k=fnm, extracted=True, postfix='full')
        txt = mc.mxl2str(path, n_bar=None)
        mic(txt)

        scr = mc.str2score(txt, omit_eos=False, title='Test')
        mic(scr)
        scr.show()
    # check_encode_decode()

    def check_broken_render():
        mc = MusicConverter(mode=md, augment_key=True)
        scr = mc.str2score(gen_broken, omit_eos=True, title='Check Broken')
        mic(scr)

        def check_notes():
            for part in scr.parts:
                bars = list(part[Measure])
                bar = bars[10]
                mic(bar)
                for e in bar:
                    if isinstance(e, Note):
                        strt, end = get_offset(e), get_end_qlen(e)
                        p = e.pitch.nameWithOctave
                        mic(e, strt, end, p)
                    else:
                        mic(e)
        # check_notes()

        def check_same_offset():
            d = dict()
            for part in scr.parts:
                d[part.partName] = [bar.offset for bar in part[Measure]]
            mic(d)
        # check_same_offset()
        scr.show()
    check_broken_render()
