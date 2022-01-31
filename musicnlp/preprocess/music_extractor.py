"""
Since Sun. Jan. 30th, an updated module for music/melody extraction, with a duration-quantized approach

See `melody_extractor` for the old version.
"""

from copy import deepcopy
from warnings import warn
from fractions import Fraction
from collections import defaultdict, Counter

from music21.stream import Score, Measure, Voice
from music21.meter import TimeSignature
from music21.tempo import MetronomeMark
from music21.note import Rest, Note
from music21.chord import Chord
from music21.duration import Duration

from musicnlp.util import *


def expand_bar(bar: Union[Measure, Voice], keep_chord=False, number=None) -> List[Union[tuple[Note], Rest, Note]]:
    """
    Expand elements in a bar into individual notes, no order is enforced

    :param bar: A music21 measure to expand
    :param keep_chord: If true, `Chord`s are not expanded
    :param number: For passing bar number recursively to Voice 

    .. note:: Triplets (potentially any n-plets) are grouped; `Voice`s are expanded
    """
    if not hasattr(expand_bar, 'post'):
        expand_bar.post = 'plet'  # Postfix for all tuplets, e.g. `Triplet`, `Quintuplet`
    if not hasattr(expand_bar, 'post2tup'):
        expand_bar.pref2n = dict(  # Tuplet prefix to the expected number of notes
            Tri=3,
            Quintu=5,
            Nonu=9
        )
    post = expand_bar.post

    lst = []
    it = iter(bar)
    elm = next(it, None)
    while elm is not None:
        if hasattr(elm, 'fullName') and post in elm.fullName:
            pref = elm.fullName[:elm.fullName.find(post)].split()[-1]
            tup = f'{pref}{post}'
            if pref in expand_bar.pref2n:
                n_tup = expand_bar.pref2n[pref]
            else:
                assert pref == 'Tu'  # A generic case, music21 processing, different from that of MuseScore
                # e.g. 'C in octave 1 Dotted 32nd Tuplet of 9/8ths (1/6 QL) Note' makes 9 notes in tuplet
                words = elm.fullName.split()
                word_n_tup = words[words.index(tup)+2]
                n_tup = int(word_n_tup[:word_n_tup.find('/')])

            elms_tup: List[Note] = [elm]
            elm_ = next(it, None)
            while elm_ is not None and tup in elm_.fullName:  # Look for all elements of the same `n_tup`
                elms_tup.append(elm_)
                elm_ = next(it, None)  # Peeked 1 ahead

            # Consecutive tuplet notes => (potentially multiple) groups
            it_tup = iter(elms_tup)
            e_tup = next(it_tup, None)
            dur: Union[Fraction, float] = 0
            idx, idx_prev, idx_last = 0, 0, len(elms_tup)-1
            n_tup_curr = 0
            trip_added = False
            idx_tup_strt = len(lst)

            def check_wrong_n_tup():
                ln = len(lst[-1])
                if ln != n_tup:
                    warn(f'Invalid {tup}: {tup} with invalid number of notes added at bar#{number}'
                         f' - expect {n_tup}, got {ln}')
            while e_tup is not None:  # MIDI & MuseScore transcription quality, e.g. A triplet may not contain 3 notes
                dur += e_tup.duration.quarterLength
                n_tup_curr += 1
                # TODO: generalize beat/tuplet duration checking logic, might involve time signature
                # Enforce a tuplet must have at least `n_tup` notes
                # Duration ends as a beat; Heuristic for end of tuplet group
                if n_tup_curr >= n_tup and dur.denominator == 1:
                    lst.append(tuple(elms_tup[idx_prev:idx+1]))
                    trip_added = True

                    # Prep for next tuplet
                    idx_prev = idx+1
                    n_tup_curr = 0
                    dur = 0

                    if idx == idx_last:  # Postpone warning later, see below
                        check_wrong_n_tup()
                # Processed til last element, last tuplet group not enough elements
                if idx == idx_last and n_tup_curr < n_tup:
                    assert trip_added
                    assert dur.denominator == 1
                    lst[-1] = lst[-1] + tuple(elms_tup[idx_prev:])  # Join the prior tuplet group
                    check_wrong_n_tup()
                idx += 1
                e_tup = next(it_tup, None)
            # All triple notes with the same `n_tup` are added
            assert sum(len(tup) for tup in lst[idx_tup_strt:]) == len(elms_tup)
            if not keep_chord:
                tups_new = []
                has_chord = False
                for i in range(idx_tup_strt, len(lst)):  # Ensure all tuplet groups contain no Chord
                    tup = lst[i]
                    # Bad transcription quality => Keep all possible tuplet combinations
                    # Expect to be the same
                    if any(isinstance(n, Chord) for n in tup):
                        has_chord = True
                        opns = [tuple(n.notes) if isinstance(n, Chord) else (n,) for n in tup]
                        tups_new.extend(list(itertools.product(*opns)))
                if has_chord:  # Update prior triplet groups
                    lst = lst[:idx_tup_strt] + tups_new
            if not trip_added:
                ic('triplet not added')
                exit(1)
            elm = elm_
            continue  # Skip `next` for peeked 1 step ahead
        elif isinstance(elm, (Note, Rest)):
            lst.append(elm)
        elif isinstance(elm, Chord):
            if keep_chord:
                lst.append(elm)
            else:
                notes = deepcopy(elm.notes)
                for n in notes:
                    n.offset += elm.offset  # Shift offset in the scope of bar
                lst.extend(notes)
        else:
            if not isinstance(elm, (  # Ensure all relevant types are considered
                TimeSignature, MetronomeMark, Voice,
                m21.layout.LayoutBase, m21.clef.Clef, m21.key.KeySignature, m21.bar.Barline
            )):
                ic(elm)
                print('unexpected type')
                exit(1)
        elm = next(it, None)
    if bar.hasVoices():  # Join all voices to notes
        lst.extend(join_its(expand_bar(v, number=bar.number) for v in bar.voices))
    return lst


class MusicExtractor:
    """
    Extract melody and potentially chords from MXL music scores => An 1D polyphonic representation
    """
    def __init__(self, scr: Union[str, Score], precision: int = 5, mode: str = 'melody'):
        """
        :param scr: A music21 Score object, or file path to an MXL file
        :param precision: Bar duration quantization, see `melody_extractor.MxlMelodyExtractor`
        :param mode: Extraction mode, one of [`melody`, `full`]
            `melody`: Only melody is extracted
            `full`: Melody and Chord as 2 separate channels extracted TODO
        """
        if isinstance(scr, str):
            self.scr = m21.converter.parse(scr)
        else:
            self.scr = scr
        self.scr: Score

        self.prec = precision
        self.mode = mode

    @staticmethod
    def it_bars(scr: Score) -> Iterator[tuple[Measure, TimeSignature, MetronomeMark]]:
        """
        Unroll a score by time, with the time signatures of each bar
        """
        time_sig, tempo = None, None
        for idx, bars in enumerate(zip(*[list(p[Measure]) for p in scr.parts])):  # Bars for all tracks across time
            assert_list_same_elms([b.number for b in bars])  # Bar numbers should be the same

            # Update time signature
            tss = [b[TimeSignature] for b in bars]
            if idx == 0 or any(tss):  # 1st bar must have time signature defined
                assert all(len(t) == 1 for t in tss)
                tss = [next(t) for t in tss]
                assert_list_same_elms([(ds.numerator, ds.denominator) for ds in tss])
                time_sig = tss[0]

            tempos = [b[MetronomeMark] for b in bars]
            if idx == 0 or any(tempos):
                tempos = [t for t in tempos if len(t) != 0]
                # When multiple tempos, take the mean
                tempos = [MetronomeMark(number=np.array([t.number for t in ts]).mean()) for ts in tempos]
                bpms = [t.number for t in tempos]
                assert_list_same_elms(bpms)

                tempo = MetronomeMark(number=bpms[0])
            yield bars, time_sig, tempo

    def __call__(self, exp='mxl'):
        scr = deepcopy(self.scr)

        lst_bar_info = list(MusicExtractor.it_bars(scr))  # TODO
        lst_notes: List[List[Union[Note, Chord, tuple[Note]]]] = []  # TODO: melody only
        for n_out, (bars, time_sig, tempo) in enumerate(lst_bar_info):
            number = bars[0].number
            ic(number)
            # if number == 73:
            #     for b in bars:
            #         b.show()
            n_slots_per_beat, n_slots = time_sig2n_slots(time_sig, self.prec)
            notes = sum((expand_bar(b, keep_chord=self.mode == 'full') for b in bars), [])

            def note2pitch(note):
                if isinstance(note, tuple):  # Triplet, return average pitch
                    # Duration for each note not necessarily same duration, for transcription quality
                    fs, durs = zip(*[(note2pitch(n__), n__.duration.quarterLength) for n__ in note])
                    return np.average(fs, weights=durs)
                elif isinstance(note, Note):
                    return note.pitch.frequency
                else:
                    assert isinstance(note, Rest)
                    return 0  # `Rest` given pitch frequency of 0

            groups = defaultdict(list)  # Group notes by starting location
            for n in notes:
                n_ = n[0] if isinstance(n, tuple) else n
                groups[n_.offset].append(n)
            groups = {offset: sorted(ns, key=note2pitch) for offset, ns in groups.items()}
            # if number == 73:
            #     ic(groups)

            def get_notes_out() -> List[Union[Note, Chord, tuple[Note]]]:
                ns_out = []
                offset_next = 0
                for offset in sorted(groups.keys()):  # Pass through notes in order
                    notes_ = groups[offset]
                    nt = notes_[-1]
                    if offset < offset_next:
                        if note2pitch(nt) > note2pitch(ns_out[-1]):
                            # Offset would closely line up across tracks, expect this to be less frequent
                            warn(f'High pitch overlap: later overlapping note with higher pitch observed '
                                 f'at bar#{number} - prior note truncated')
                            # if number == 73:
                            #     ic(nt, ns_out[-1])
                            if isinstance(ns_out[-1], tuple):  # TODO: recomputing notes, if triplet is overlapping
                                ic('triplet being truncated')
                                exit(1)
                            else:  # Triplet replaces pr..           ior note, which is definitely non triplet
                                nt_ = nt[0] if isinstance(nt, tuple) else nt  # Definitely non-0 for offset grouping
                                ns_out[-1].duration = Duration(quarterLength=nt_.offset - ns_out[-1].offset)
                        else:  # Skip if later note is lower in pitch
                            continue
                    ns_out.append(nt)  # Note with the highest pitch
                    nt_ = nt[-1] if isinstance(nt, tuple) else nt
                    offset_next = nt_.offset + nt_.duration.quarterLength
                return ns_out
            notes_out = get_notes_out()
            assert_notes_no_overlap(notes_out)  # Ensure notes cover the entire bar
            n_last = notes_out[-1]
            n_last = n_last[-1] if isinstance(n_last, tuple) else n_last
            assert (n_last.offset + n_last.duration.quarterLength) == (time_sig.numerator / time_sig.denominator * 4)
            # if number == 73:
            #     for n in notes_out:
            #         if isinstance(n, tuple):
            #             ic(n)
            #         else:
            #             ic(n, n.offset, n.duration.quarterLength)
            #     exit(1)
            lst_notes.append(notes_out)

        # for p in scr.parts:
        #     ic(p.id, p.partName)
        # exit(1)
        tempo_nums, time_sigs, bars = zip(*[  # Pick 1st bar arbitrarily
            (tempo.number, time_sig, bars[0].duration.quarterLength) for bars, time_sig, tempo in lst_bar_info
        ])
        # ic(tempo_nums, time_sigs, bars)
        mean_tempo = np.array(tempo_nums).mean()
        counter_ts = Counter((ts.numerator, ts.denominator) for ts in time_sigs)
        time_sig_mode = max(counter_ts, key=counter_ts.get)
        th = 0.95
        n_mode, n_bar = counter_ts[time_sig_mode], len(time_sigs)
        if n_mode / n_bar < th:  # Arbitrary threshold; Too much invalid time signature
            warn(f'Inconsistent Time Signatures: ratio of mode time signature below {th}'
                 f' - #mode {n_mode}, #total {n_bar}')
        # exit(1)
        # ic(vars(scr.metadata))
        # ic(scr.metadata.all())
        # ic(scr.metadata.title)
        # exit(1)

        if exp == 'mxl':
            lst_bars = []
            for i, notes in enumerate(lst_notes):
                bar = Measure(number=i)  # Original bar number may not start from 0
                for n in notes:
                    bar.append(n)  # So that works with Tuplets
                # if i == 73:
                #     for e in bar:
                #         ic(e, e.offset, e.duration.quarterLength)
                # ic(bar)
                # for e in bar:
                #     ic(e, e.offset, e.duration.quarterLength)
                lst_bars.append(bar)

            scr_out = Score()
            scr_out.insert(m21.metadata.Metadata())
            title = scr.metadata.title
            if title.endswith('.mxl'):
                title = title[:-4]
            post = 'Melody only' if self.mode == 'melody' else 'Melody & Chord'
            title = f'{title}, {post}'
            scr_out.metadata.title = title
            scr_out.metadata.composer = PKG_NM

            part_nm = 'Melody, Ch#1'  # TODO: a 2nd chord part
            part = m21.stream.Part(partName=part_nm)
            part.partName = part_nm
            # ic(part.id, part.partName)
            instr = m21.instrument.Piano()
            part.append(instr)
            # ic(lst_bars)
            part.append(lst_bars)
            # for e in part:
            #     ic(e)
            #     if isinstance(e, Measure):
            #         ic(e.number)

            bar0 = part.measure(0)  # Insert metadata into 1st bar
            bar0.insert(MetronomeMark(number=mean_tempo))
            bar0.insert(TimeSignature(f'{time_sig_mode[0]}/{time_sig_mode[1]}'))

            dir_nm = config(f'{DIR_DSET}.MXL_EG.dir_nm')
            dir_nm = f'{dir_nm}_out'
            scr_out.append(part)
            # scr.show()
            scr_out.write(fmt='mxl', fp=os.path.join(PATH_BASE, DIR_DSET, dir_nm, f'{title}.mxl'))


            # scr = Score(id='mainScore')
            # p0 = stream.Part(id='part0')
            # p1 = stream.Part(id='part1')
            #
            # m01 = stream.Measure(number=1)
            # m01.append(note.Note('C', type="whole"))
            # m02 = stream.Measure(number=2)
            # m02.append(note.Note('D', type="whole"))
            # p0.append([m01, m02])
            #
            # m11 = stream.Measure(number=1)
            # m11.append(note.Note('E', type="whole"))
            # m12 = stream.Measure(number=2)
            # m12.append(note.Note('F', type="whole"))
            # p1.append([m11, m12])
            #
            # s.insert(0, p0)
            # s.insert(0, p1)
            # s.show('text')


            # # Pick a `Part` to replace elements one by one, the 1st part selected as it contains all metadata
            # idx_part = 0
            # scr.remove(list(filter(lambda p: p is not scr.parts[idx_part], scr.parts)))
            # assert len(scr.parts) == 1
            # part = scr.parts[0]
            # pnm = part.partName
            #
            # def pnm_ori(nm):
            #     return nm[:nm.rfind(', CH #')]
            #
            # pnm_ori_ = pnm_ori(pnm)
            # for p in self.scr.parts[1:]:
            #     # There should be no tempo in all other channels, unless essentially the "same" channel
            #     assert len(p[m21.tempo.MetronomeMark]) == 0 or pnm_ori(p.partName) == pnm_ori_
            #
            # vbs = self.vertical_bars(self.scr)
            # if self.verbose:
            #     print(f'{now()}| Extracting music [{stem(self.fnm)}] of duration [{self.score_seconds(vbs)}]... ')
            # for idx, bar in enumerate(part[m21.stream.Measure]):  # Ensure each bar is set
            #     vb = vbs[idx].single()
            #     pnm_ = vb.pnm_with_max_pitch(method='fqs')
            #     assert bar.number == idx + self.bar_strt_idx
            #     assert part.index(bar) == idx + 1
            #     part.replace(bar, vb.bars[pnm_])
            #
            # # Set instrument as Piano
            # instr = m21.instrument.Piano()
            # [part.remove(ins) for ins in part[m21.instrument.Instrument]]
            # part.insert(instr)
            # part.partName = f'{PKG_NM}, {instr.instrumentName}, CH #1'
            #
            # # Set tempo
            # [bar.removeByClass(m21.tempo.MetronomeMark) for bar in part[m21.stream.Measure]]
            # self.tempo_strt.number = self.mean_tempo
            # bar0 = part.measure(self.bar_strt_idx)
            # bar0.insert(self.tempo_strt)
            #
            # title = f'{self.score_title}, bar with max pitch'
            # if exp == 'mxl':
            #     dir_nm = config(f'{DIR_DSET}.MXL_EG.dir_nm')
            #     dir_nm = f'{dir_nm}_out'
            #     scr.write(fmt='mxl', fp=os.path.join(PATH_BASE, DIR_DSET, dir_nm, f'{title}.mxl'))
            # elif exp == 'symbol':
            #     # Get time signature for each bar
            #     lst_bar_n_ts = bars2lst_bar_n_ts(part[m21.stream.Measure])
            #     return self.tokenizer(lst_bar_n_ts)
            # else:
            #     return scr


if __name__ == '__main__':
    from icecream import ic

    def toy_example():
        fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        # fnm = eg_songs('Shape of You', fmt='MXL')
        ic(fnm)
        me = MusicExtractor(fnm)
        me(exp='mxl')
    toy_example()
