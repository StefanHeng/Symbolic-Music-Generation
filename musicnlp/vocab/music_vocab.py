from enum import Enum
from typing import Set, Optional

from music21.pitch import Pitch

from musicnlp.util.music_lib import *


class VocabType(Enum):
    time_sig, tempo, duration, pitch, special = list(range(5))

    @classmethod
    def compact(cls) -> Iterator['VocabType']:
        """
        :return: Iterator of all token types with compact representation
        """
        for i in range(4):
            yield cls(i)


Compact = Union[TsTup, int, Dur]


class MusicVocabulary:
    """
    Stores mapping between string tokens and integer ids
    & support the conversion, from relevant `music21` objects to [`str`, `int] conversion
    """
    start_of_bar = '<bar>'
    end_of_song = '</s>'
    start_of_tuplet = '<tup>'
    end_of_tuplet = '</tup>'

    SPEC_TOKS = dict(
        sep='_',  # Separation
        rest='r',
        prefix_pitch='p',
        prefix_duration='d',
        start_of_tuplet=start_of_tuplet,
        end_of_tuplet=end_of_tuplet,
        start_of_bar=start_of_bar,
        end_of_song=end_of_song,
        prefix_time_sig='TimeSig',
        prefix_tempo='Tempo'
    )
    # Uncommon Time Signatures in music theory, but empirically seen in MIDI data
    UNCOM_TSS: List[TsTup] = [(1, 4)]

    RE_INT = r'[1-9]\d*'
    RE1 = rf'(?P<num>{RE_INT})'
    RE2 = rf'(?P<numer>{RE_INT})/(?P<denom>{RE_INT})'

    def __init__(self, prec: int = 5, color: bool = False):
        """
        :param prec: See `MusicTokenizer`
        :param color: If True, string outputs are colorized
            Update individual coloring of subsequent tokens via `__getitem__`
        """
        self.prec = prec
        self.color = color

        specs = MusicVocabulary.SPEC_TOKS  # Syntactic sugar
        sep = specs['sep']
        self.cache = dict(  # Common intermediary substrings
            pref_dur=specs['prefix_duration']+sep,
            pref_pch=specs['prefix_pitch']+sep,
            pref_time_sig=specs['prefix_time_sig']+sep,
            pref_tempo=specs['prefix_tempo']+sep,
            bot=self.__getitem__('start_of_tuplet'),
            eot=self.__getitem__('end_of_tuplet')
        )
        self.rest = self.cache['rest'] = self.cache['pref_pch'] + specs['rest']

        self.type2compact_re = {
            VocabType.duration: dict(
                int=re.compile(rf'^{self.cache["pref_dur"]}{MusicVocabulary.RE1}$'),
                frac=re.compile(rf'^{self.cache["pref_dur"]}{MusicVocabulary.RE2}$'),
            ),
            VocabType.pitch: re.compile(rf'^{self.cache["pref_pch"]}{MusicVocabulary.RE2}$'),
            VocabType.time_sig: re.compile(rf'^{self.cache["pref_time_sig"]}{MusicVocabulary.RE2}$'),
            VocabType.tempo: re.compile(rf'^{self.cache["pref_tempo"]}{MusicVocabulary.RE1}$')
        }

        self.compacts: Set[VocabType] = set(VocabType.compact())

        def elm2str(elm):
            return self.__call__(elm, color=False, return_int=False)
        # self.n_slots = OrderedDict([  # Reserved slots for each token category
        #     ('special', 32),
        #     ('time_sig', 32),  # A few common time signatures only
        #     ('tempo', 256),  # Usually range from 20+ to 200+
        #     # 128 pitches in MIDI representation; TODO: with music-theory, mod-7 scale, may increase
        #     ('pitch', 256),
        #     ('duration', 256)  # Depends on quantization
        # ])

        def rev(time_sig):
            return tuple(reversed(time_sig))  # Syntactic sugar
        tss = [elm2str(rev(ts))[0] for ts in sorted(rev(ts) for ts in COMMON_TIME_SIGS + MusicVocabulary.UNCOM_TSS)]
        tempos = [elm2str(tp)[0] for tp in range(20, 220)]  # Expected normal song ranges
        pitches = [self.cache['rest']] + [self._note2pch_str(Pitch(midi=i)) for i in range(128)]

        self.toks: Dict[str, List[str]] = dict(
            special=[specs[k] for k in ('end_of_song', 'start_of_bar', 'start_of_tuplet', 'end_of_tuplet')],
            time_sig=tss,
            tempo=tempos,
            pitch=pitches,
            duration=self.get_durations(exp='str')
        )
        self.enc: Dict[str, int] = {  # Back2back index as ids
            tok: id_ for id_, tok in enumerate(join_its(toks for toks in self.toks.values()))
        }
        self.dec = {v: k for k, v in self.enc.items()}
        assert len(self.enc) == len(self.dec)  # Sanity check: no id collision

    def to_dict(self, save=False):
        d_out = dict(
            precision=self.prec,
            special_tokens={
                'start_of_bar': MusicVocabulary.start_of_bar,
                'end_of_song': MusicVocabulary.end_of_song,
                'start_of_tuplet': MusicVocabulary.start_of_tuplet,
                'end_of_tuplet': MusicVocabulary.end_of_tuplet
            },
            vocabulary=self.enc,
            n_vocabulary=len(self.enc),
        )
        if save:
            fnm = f'{self.__class__.__qualname__}, n={len(self.enc)}, prec={self.prec}, {now(for_path=True)}'
            path = os.path.join(get_processed_path(), f'{fnm}.json')
            with open(path, 'w') as f:
                json.dump(d_out, f, indent=4)
        return d_out

    def get_durations(self, bound: int = None, exp: str = 'str') -> Union[List[str], List[Dur]]:
        """
        :param bound: The upper bound for duration within a bar, in terms of `quarterLength`
            By default, support duration up to **6**
        :param exp: return type, one of ['str`, `dur`]
            If str, returns duration token
            If dur, returns `Dur`
        :return: List of durations
        """
        if bound is None:
            # TODO: support for longer duration needed?
            bound = max(ts[0]/ts[1] for ts in COMMON_TIME_SIGS + MusicVocabulary.UNCOM_TSS) * 4
            assert bound.is_integer()
        dur_slot, denom = 4/2**self.prec, 2**self.prec/4
        assert denom.is_integer()
        denom = int(denom)
        gen_numers = range(math.ceil(bound / dur_slot))
        if exp == 'str':
            return [self._note2dur_str((i+1) * dur_slot) for i in gen_numers]
        else:
            assert exp == 'dur'
            ret = [Fraction(i+1, denom) for i in gen_numers]
            return [int(f) if f.denominator == 1 else f for f in ret]

    def __len__(self):
        return len(self.enc)

    def has_compact(self, tok: str) -> bool:
        return self.type(tok) != VocabType.special

    def type(self, tok: str) -> VocabType:
        if self.cache['pref_dur'] in tok:
            return VocabType.duration
        elif self.cache['pref_pch'] in tok:
            return VocabType.pitch
        elif self.cache['pref_time_sig'] in tok:
            return VocabType.time_sig
        elif self.cache['pref_tempo'] in tok:
            return VocabType.tempo
        else:
            return VocabType.special

    @staticmethod
    def _get_group1(tok, tpl) -> int:
        return int(tpl.match(tok).group('num'))

    @staticmethod
    def _get_group2(tok, tpl) -> Tuple[int, int]:
        m = tpl.match(tok)
        return int(m.group('numer')), int(m.group('denom'))

    def compact(self, tok: str) -> Compact:
        """
        Convert tokens to the numeric format

        More compact, intended for statistics

        Raise error is special tokens passed

        :return: If time signature, returns 2-tuple of (int, int),
            If tempo, returns integer of tempo number
            If pitch, returns the pitch MIDI number
            If duration, returns the duration quarterLength
        """
        assert self.has_compact(tok), ValueError(f'{logi(tok)} does not have a compact representation')
        typ = self.type(tok)
        tpl = self.type2compact_re[typ]
        if typ == VocabType.duration:
            if '/' in tok:
                numer, denom = MusicVocabulary._get_group2(tok, tpl['frac'])
                assert math.log2(denom).is_integer()
                # Quantized so definitely an exact float, but keep Fraction for exact additions
                return Fraction(numer, denom)
            else:
                return MusicVocabulary._get_group1(tok, tpl['int'])
        elif typ == VocabType.pitch:
            if tok == self.cache['rest']:
                return -1
            else:
                pch, octave = MusicVocabulary._get_group2(tok, tpl)
                return pch-1 + octave*12  # See `pch2step`
        elif typ == VocabType.time_sig:
            return MusicVocabulary._get_group2(tok, tpl)
        else:  # VocabType.tempo
            return MusicVocabulary._get_group1(tok, tpl)

    def uncompact(self, type: VocabType, compact: Optional[Compact] = None) -> str:
        """
        Reverse operation of `compact`, returns the music "decoded" string
        """
        assert type != VocabType.special, ValueError(f'Compact representation for special types not supported')
        if type == VocabType.duration:
            assert isinstance(compact, (int, Tuple[int, int]))
            if isinstance(compact, int):
                return f'{self.cache["pref_dur"]}{compact}'
            else:
                return f'{self.cache["pref_dur"]}{compact[0]}/{compact[1]}'
        elif type == VocabType.pitch:
            assert isinstance(compact, int)
            if compact == -1:
                return self.cache['rest']
            else:
                pch, octave = compact % 12, compact // 12
                return f'{self.cache["pref_pch"]}{pch}/{octave}'
        elif type == VocabType.time_sig:
            assert isinstance(compact, tuple)
            return f'{self.cache["pref_time_sig"]}{compact[0]}/{compact[1]}'
        else:  # VocabType.tempo
            assert isinstance(compact, int)
            return f'{self.cache["pref_tempo"]}{compact}'

    @staticmethod
    def pitch_midi2name(midi: int) -> str:
        if midi == -1:
            return 'rest'
        else:
            pch = m21.pitch.Pitch(midi=midi)
            return f'{pch.name}/{pch.octave}'

    def _colorize_spec(self, s: str, color: bool = None) -> str:
        c = self.color if color is None else color
        return log_s(s, c='m') if c else s

    def __getitem__(self, k: str) -> str:
        """
        Index into the special tokens
        """
        return self._colorize_spec(MusicVocabulary.SPEC_TOKS[k])

    def __call__(
            self, elm: Union[ExtNote, Union[TimeSignature, TsTup], Union[MetronomeMark, int]],
            color: bool = None,
            return_int: bool = False  # TODO
    ) -> Union[List[str], List[int]]:  # TODO: Support chords?
        """
        Convert music21 element to string or int

        :param elm: A relevant token in melody extraction
        :param color: If given, overrides coloring for current call
        :param return_int: If true, integer ids are returned
        :return List of strings of the converted tokens
        """
        c = self.color if color is None else color

        def colorize(s):
            return self._colorize_spec(s, color=c)

        if isinstance(elm, TimeSignature) or (isinstance(elm, tuple) and isinstance(elm[0], int)):  # Time Signature
            if isinstance(elm, TimeSignature):
                top, bot = elm.numerator, elm.denominator
            else:
                top, bot = elm
            return [colorize(self.cache['pref_time_sig']+f'{top}/{bot}')]
        elif isinstance(elm, (int, MetronomeMark)):  # Tempo
            if isinstance(elm, MetronomeMark):
                elm = round(elm.number)  # should be integer
            return [colorize(self.cache['pref_tempo']+str(elm))]
        elif isinstance(elm, Rest):
            r = self.cache['rest']
            return [log_s(r, c='b') if color else r, self._note2dur_str(elm)]
        elif isinstance(elm, Note):
            return [self._note2pch_str(elm), self._note2dur_str(elm)]
        elif isinstance(elm, tuple):
            # Sum duration for all tuplets
            bot, eot = self.cache['bot'], self.cache['eot']
            return [colorize(bot)] + [
                (self._note2pch_str(e)) for e in elm
            ] + [self._note2dur_str(elm)] + [colorize(eot)]
        else:  # TODO: chords
            ic('other element type', elm)
            exit(1)

    def _note2dur_str(
            self, e: Union[ExtNote, Dur]) -> str:
        """
        :param e: A note, tuplet, or a numeric representing duration
        """
        # If a float, expect multiple of powers of 2
        dur = Fraction(e if isinstance(e, (float, Fraction)) else note2dur(e))
        if dur.denominator == 1:
            s = f'{self.cache["pref_dur"]}{dur.numerator}'
        else:
            s = f'{self.cache["pref_dur"]}{dur.numerator}/{dur.denominator}'
        return log_s(s, c='g') if self.color else s

    def _note2pch_str(self, note: Union[Note, Rest, Pitch]) -> str:
        """
        :param note: A note, tuplet, or a music21.pitch.Pitch
        """
        def pch2step(p: Pitch) -> int:
            """
            Naive mapping to the physical, mod-12 pitch frequency, in [1-12]
            """
            return (p.midi % 12) + 1
        if isinstance(note, Rest):
            s = self.cache["rest"]
        else:
            pitch = note.pitch if isinstance(note, Note) else note
            # `pitch.name` follows certain scale by music21 default, may cause confusion
            s = f'{self.cache["pref_pch"]}{pch2step(pitch)}/{pitch.octave}'
        return log_s(s, c='b') if self.color else s

    def t2i(self, tok):
        return self.enc[tok]

    def i2t(self, id_):
        return self.dec[id_]

    def encode(self, s: Union[str, List[str], List[List[str]]]) -> Union[int, List[int], List[List[int]]]:
        """
        Convert string token or tokens to integer id
        """
        if isinstance(s, List) and isinstance(s[0], List):
            return list(conc_map(self.encode, s))
        elif isinstance(s, List):
            return [self.enc[s_] for s_ in s]
        else:
            return self.enc[s]

    def decode(self, id_: Union[int, List[int], List[List[int]]]) -> Union[str, List[str], List[List[str]]]:
        """
        Reverse function of `str2id`
        """
        if isinstance(id_, List) and isinstance(id_[0], List):
            return list(conc_map(self.decode, id_))
        elif isinstance(id_, List):
            return [self.dec[i_] for i_ in id_]
        else:
            return self.dec[id_]


if __name__ == '__main__':
    from icecream import ic

    mv = MusicVocabulary()
    # ic(mv.get_durations(exp='dur'))

    ic(mv.to_dict(save=True))
