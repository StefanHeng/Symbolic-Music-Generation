"""
Music preprocessing utilities
"""

import math
from copy import deepcopy
from typing import Iterator
from fractions import Fraction

import music21 as m21
from music21.meter import TimeSignature
from music21.note import Note, Rest
from music21.chord import Chord

from .util import *


KEEP_OBSOLETE = False
if KEEP_OBSOLETE:
    import mido
    from mido import MidiFile
    import pretty_midi
    from pretty_midi import PrettyMIDI
    import librosa
    from librosa import display


ExtNote = Union[Note, Rest, Tuple[Union[Note, Rest]]]  # Note entity/group as far as music extraction is concerned
SNote = Union[Note, Rest]  # Single note
Dur = Union[float, Fraction]
TsTup = Tuple[int, int]


def time_sig2n_slots(time_sig: m21.meter.TimeSignature, precision: int) -> Tuple[int, int]:
    """
    :return: 2 tuple of (#time slots per beat, #time slots in total)
    """
    denom = time_sig.denominator
    numer = time_sig.numerator
    n_slots_per_beat = (1 / denom / (2 ** -precision))
    assert n_slots_per_beat.is_integer()
    n_slots_per_beat = int(n_slots_per_beat)
    n_slots = int(numer * n_slots_per_beat)
    return n_slots_per_beat, n_slots


def it_m21_elm(
        stream: Union[m21.stream.Measure, m21.stream.Part, m21.stream.Score, m21.stream.Voice],
        types=(Note, Rest)
):
    """
    Iterates elements in a stream, for those that are instances of that of `type`, in the original order
    """
    if isinstance(stream, (m21.stream.Measure, m21.stream.Voice)):
        return iter(filter(lambda elm: isinstance(elm, types), stream))
    else:
        return iter(filter(lambda elm: isinstance(elm, types), stream.flatten()))


def group_triplets(bar) -> List[ExtNote]:
    """
    Identify triplets from a bar from normal notes & group them

    Expect no `Chord` or `Voice` in bar
    """
    lst = []
    it = it_m21_elm(bar)
    elm = next(it, None)
    while elm:
        if 'Triplet' in elm.fullName:
            elm2, elm3 = next(it, None), next(it, None)
            assert elm2 is not None and elm3 is not None
            assert 'Triplet' in elm2.fullName and 'Triplet' in elm3.fullName
            lst.append((elm, elm2, elm3))
        else:
            lst.append(elm)
        elm = next(it, None)
    return lst


EPS = 1e-6


def is_int(num: Union[float, Fraction], check_close: Union[bool, float] = True) -> bool:
    if isinstance(num, float):
        if check_close:  # Numeric issue summing Fractions with floats
            eps = check_close if isinstance(check_close, float) else 1e-6
            return math.isclose(num, round(num), abs_tol=eps)
        else:
            return num.is_integer()
    else:
        return num.denominator == 1


def is_8th(d: Dur) -> bool:
    """
    :return If Duration `d` in quarterLength, is multiple of 8th note
    """
    return is_int(d*2)


COMMON_TIME_SIGS: List[TsTup] = sorted(  # Sort first by denominator
    [(4, 4), (2, 4), (2, 2), (3, 4), (6, 8), (5, 4), (12, 8)],
    key=lambda tup_: tuple(reversed(tup_))
)


def is_common_time_sig(ts: Union[TimeSignature, TsTup]):
    if not hasattr(is_common_time_sig, 'COM_TS'):  # List of common time signatures
        is_common_time_sig.COM_TS = set(COMMON_TIME_SIGS)
    if isinstance(ts, TimeSignature):
        ts = (ts.numerator, ts.denominator)
    return ts in is_common_time_sig.COM_TS


def note2pitch(note: ExtNote):
    if isinstance(note, tuple):  # Triplet, return average pitch
        # Duration for each note not necessarily same duration, for transcription quality
        fs, durs = zip(*[(note2pitch(n__), n__.duration.quarterLength) for n__ in note])
        return np.average(fs, weights=durs)
    elif isinstance(note, Note):
        return note.pitch.frequency
    else:
        assert isinstance(note, Rest)
        return 0  # `Rest` given pitch frequency of 0


def note2dur(note: ExtNote) -> Dur:
    if isinstance(note, tuple):
        return sum(note2dur(nt) for nt in note)
    else:
        return note.duration.quarterLength


def notes2offset_duration(notes: Union[List[ExtNote], ExtNote]) -> Tuple[List[float], List[Dur]]:
    if isinstance(notes, list):  # Else, single tuplet notes
        notes = flatten_notes(unroll_notes(notes))
    offsets, durs = zip(*[(n.offset, n.duration.quarterLength) for n in notes])
    return offsets, durs


def flatten_notes(notes: Iterable[ExtNote]) -> Iterator[SNote]:
    """
    Expand the intermediate grouping of tuplets
    """
    for n in notes:
        if isinstance(n, tuple):
            for n_ in n:
                yield n_
        else:
            yield n


def unpack_notes(
        notes: List[ExtNote]
) -> Tuple[List[SNote], object]:
    """
    :param notes: Notes representation with tuplets in tuples
    :return: 2-tuple of flattened notes, and a cache for reconstructing the packed tuple representation

    .. note:: See `pack_notes`
    """
    return (
        list(flatten_notes(notes)),
        # Stores the starting index & length of each tuple
        {idx: len(notes[idx]) for idx in range(len(notes)) if isinstance(notes[idx], tuple)}
    )


def pack_notes(notes: List[SNote], cache: object) -> List[ExtNote]:
    """
    Reconstructs original note with tuplets in tuples from cache
    """
    notes_out = []
    cache: Dict[int, int]
    if cache:
        it = iter(notes)
        idx = 0
        n_notes = len(notes)
        while idx < n_notes:
            if idx in cache:
                tups = []
                for _ in range(cache[idx]):
                    tups.append(next(it))
                    idx += 1
                notes_out.append(tuple(tups))
            else:
                notes_out.append(next(it))
                idx += 1
        return notes_out
    else:  # Cache empty, i.e. no tuplets
        return notes


def unroll_notes(notes: List[ExtNote]) -> List[ExtNote]:
    """
    :param notes: individual notes with offsets not back-to-back
    :return: Notes as if jointed in time together

    .. Original notes unmodified
    """
    # if not isinstance(notes, list):
    #     notes = list(notes)
    # notes[0].offset = 0
    if is_notes_no_overlap(notes):
        return notes
    else:
        notes_ = list(flatten_notes(notes))
        notes_ = [deepcopy(n) for n in notes_]
        offsets = [0]
        strt = notes_[0].duration.quarterLength
        for note in notes_[1:]:
            offsets.append(strt)
            strt += note.duration.quarterLength
        offsets = iter(offsets)
        for i, note in enumerate(notes):
            if isinstance(note, tuple):
                notes_tup: List[SNote] = list(note)
                for idx, n in enumerate(notes_tup):
                    # If `Chord`, notes inside with offset of 0 always, leave unchanged
                    notes_tup[idx].offset = next(offsets)
                notes[i] = tuple(notes_tup)
            else:
                notes[i].offset = next(offsets)
        return notes


def quarter_len2fraction(q_len: Dur) -> Fraction:
    """
    :param q_len: A quarterLength value to convert
        Requires one of power of 2
    """
    if isinstance(q_len, float):
        numer, denom = q_len, 1.
        while not (numer.is_integer() and denom.is_integer()):  # Should terminate quick for the expected use case
            numer *= 2
            denom *= 2
        return Fraction(int(numer), int(denom))
    else:
        return q_len


def note2note_cleaned(note: ExtNote, q_len=None, offset=None) -> ExtNote:
    """
    :return: A cleaned version of Note or tuplets with only duration, offset and pitch set
        Notes in tuplets are set with-equal duration given by (`q_len` if `q_len` given, else tuplet total length)
    """
    if q_len is None:
        q_len = note2dur(note)
    if isinstance(note, tuple):
        offset = offset if offset is not None else note[0].offset
        dur_ea = quarter_len2fraction(q_len)/len(note)
        notes: List[SNote] = [note2note_cleaned(n, q_len=dur_ea) for n in note]
        for i, nt_tup in enumerate(notes):
            notes[i].offset = offset + dur_ea * i
        return tuple(notes)
    dur = m21.duration.Duration(quarterLength=q_len)
    if isinstance(note, Note):  # Removes e.g. `tie`s
        nt = Note(pitch=m21.pitch.Pitch(midi=note.pitch.midi), duration=dur)
        # Setting offset in constructor doesn't seem to work per `music21
        nt.offset = offset if offset is not None else note.offset
        return nt
    elif isinstance(note, Rest):
        nt = Rest(duration=dur, offset=note.offset)
        nt.offset = offset if offset is not None else note.offset
        return nt
    else:
        assert isinstance(note, Chord)  # TODO
        print('clean chord')
        exit(1)


def is_notes_no_overlap(notes: Iterable[ExtNote]) -> bool:
    """
    :return True if notes don't overlap, given the start time and duration
    """
    notes = flatten_notes(notes)
    note = next(notes, None)
    end = note.offset + note.duration.quarterLength
    note = next(notes, None)
    while note is not None:
        # Current note should begin, after the previous one ends
        # Since numeric representation of one-third durations, aka tuplets
        if (end-EPS) <= note.offset:
            end = note.offset + note.duration.quarterLength
            note = next(notes, None)
        else:
            return False
    return True


def is_valid_bar_notes(notes: Iterable[ExtNote], time_sig: TimeSignature) -> bool:
    dur_bar = time_sig.numerator / time_sig.denominator * 4
    # Ensure notes cover the entire bar; For addition between `float`s and `Fraction`s
    return is_notes_no_overlap(notes) \
        and math.isclose(sum(n.duration.quarterLength for n in flatten_notes(notes)), dur_bar, abs_tol=1e-6)


DEF_TPO = int(5e5)  # Midi default tempo (ms per beat, i.e. 120 BPM)
N_NOTE_OCT = 12  # Number of notes in an octave


def tempo2bpm(tempo):
    """
    :param tempo: ms per beat
    :return: Beat per minute
    """
    return 60 * 1e6 / tempo


if KEEP_OBSOLETE:
    class MidoUtil:
        def __init__(self):
            pass

        @staticmethod
        def get_msgs_by_type(fl, t_, as_dict=False):
            def _get(track):
                return list(filter(lambda tr: tr.type == t_, track))
            if type(fl) is MidiFile or type(fl) is str:
                if type(fl) is str:
                    fl = MidiFile(fl)
                d_msgs = {i: _get(trk) for i, trk in enumerate(fl.tracks)}
                return d_msgs if as_dict else flatten(d_msgs.values())
            elif type(fl) is mido.MidiTrack:
                return _get(fl)

        @staticmethod
        def get_tempo_changes(mf, dedupe=False):
            lst_msgs = MidoUtil.get_msgs_by_type(mf, 'set_tempo')
            if lst_msgs:
                tempos = [msg.tempo for msg in lst_msgs]
                return list(set(tempos)) if dedupe else tempos
            else:
                return [DEF_TPO]


    class PrettyMidiUtil:
        def __init__(self):
            pass

        @staticmethod
        def plot_single_instrument(instr_, cols=('start', 'end', 'pitch', 'velocity'), n=None):
            """
            Plots a single `pretty_midi.Instrument` channel

            :return: pd.DataFrame of respective attributes in the note, up until `n`
            """
            def _get_get(note):
                return [getattr(note, attr) for attr in cols]
            df = pd.DataFrame([_get_get(n) for n in instr_.notes], columns=cols)[:n]
            df.plot(
                figsize=(16, 9),
                lw=0.25, ms=0.3,
                title=f'Instrument notes plot - [{",".join(cols)}]'
            )
            return df

        @staticmethod
        def get_pitch_range(pm_, clip_=False):
            """
            :return: Inclusive lower and upper bound of pitch in PrettyMIDI
            """
            def _get_pitch_range():
                def _get(instr_):
                    arr = np.array([n.pitch for n in instr_.notes])
                    return np.array([arr.min(), arr.max()])

                if type(pm_) is PrettyMIDI:
                    ranges = np.vstack([_get(i) for i in pm_.instruments])
                    return ranges[:, 0].min(), ranges[:, 1].max()
                else:
                    return list(_get(pm_))
            strt, end = _get_pitch_range()
            if clip_:
                strt = math.floor(strt / N_NOTE_OCT) * N_NOTE_OCT
                end = math.ceil(end / N_NOTE_OCT) * N_NOTE_OCT
            return strt, end

        @staticmethod
        def plot_piano_roll(fl, strt=None, end=None, fqs=100, with_beats=True):
            """
            :param fl: A PrettyMIDI, Instrument, or file path string
            :param strt: Lowest pitch to plot inclusive, inferred if not given
            :param end: Highest pitch to plot inclusive, inferred if not given
            :param fqs: Sample frequency
            :param with_beats: If true, beat times are shown as vertical lines
                If fl is Instrument, expect the times needs to be provided
            """
            if type(fl) is str:
                fl = PrettyMIDI(fl)
            if strt is None and end is None:
                strt, end = PrettyMidiUtil.get_pitch_range(fl, clip_=True)
                end += 1  # inclusive np array slicing
            pr_ = fl.get_piano_roll(fqs)[strt:end]
            strt_ = pretty_midi.note_number_to_name(strt)
            end_ = pretty_midi.note_number_to_name(end)

            plt.figure(figsize=(16, (9 * (end-strt+1) / 128) + 1), constrained_layout=True)
            kwargs = dict(
                fmin=pretty_midi.note_number_to_hz(strt) if strt else None
            )
            with sns.axes_style('ticks'):
                librosa.display.specshow(
                    pr_,
                    hop_length=1, sr=fqs, x_axis='time', y_axis='cqt_note',
                    cmap='mako',
                    **kwargs
                )
            nms = np.arange(strt, end, 6)
            plt.colorbar(pad=2**(-5))
            plt.yticks(
                list(map(pretty_midi.note_number_to_hz, nms)),
                list(map(pretty_midi.note_number_to_name, nms))
            )
            if (type(with_beats) is bool and with_beats) or with_beats is not None:
                ticks = plt.yticks()[0]
                x = fl.get_beats() if type(fl) is PrettyMIDI else with_beats
                ax = plt.gca()
                ax.vlines(x=x, ymin=ticks.min(), ymax=ticks.max(), lw=0.25, alpha=0.5, label='Beats')

            plt.title(f'Piano roll plot - [{strt_}, {end_}]')
            plt.show()


    class Music21Util:
        def __init__(self):
            pass

        @staticmethod
        def plot_piano_roll(stream, title=None, s=0, e=None):
            """
            :return: music21 graph object for plotting
            """
            mess = stream.measures(s, e)
            plt_ = m21.graph.plot.HorizontalBarPitchSpaceOffset(
                mess, figsize=(16, 9), constrained_layout=False, doneAction=None
            )
            plt_.colors = sns.color_palette(palette='husl', n_colors=2**4)
            plt_.fontFamily = 'sans-serif'
            plt_.run()

            plt.tight_layout()
            x_ticks = plt.xticks()[0]
            y_ticks = plt.yticks()[0]
            x_d, y_d = np.diff(x_ticks).max(), np.diff(y_ticks).max()
            offset_x, offset_y = x_d / 2**3, y_d / 2**1
            x_s, x_e = x_ticks[0], x_ticks[-1] + x_d
            y_s, y_e = y_ticks[0], y_ticks[-1]
            plt.xlim([x_s - offset_x, x_e + offset_x])
            plt.ylim([y_s - offset_y, y_e + offset_y])

            fig = plt.gcf()
            fig.set_size_inches(16, clip(9 * (y_e - y_s) / (x_e - x_s), 16 / 2 ** 3, 16 / 2))
            strt = s
            end = len(stream.measures(0, e)) if e is None else e
            title = (
                    title or
                    (stream.metadata and stream.metadata.title) or
                    (stream.activeSite and stream.activeSite.metadata and stream.activeSite.metadata.title) or
                    stream.id
            )
            if isinstance(stream, m21.stream.Part) and stream.partName:
                title = f'{title}, {stream.partName}'
            plt.title(f'Piano roll, {title}, bars {strt}-{end}')
            txt_prop = dict(fontname='DejaVu Sans')
            plt.yticks(*plt.yticks(), **txt_prop)
            plt.show()
            return plt_


if __name__ == '__main__':
    from music21 import graph
    from icecream import ic

    # ic(tempo2bpm(DEF_TPO))

    def check_note2hz():
        for n in np.arange(0, 12*10, 12):
            ic(n, pretty_midi.note_number_to_hz(n))
    # check_note2hz()

    def check_piano_roll():
        # pm = pretty_midi.PrettyMIDI(eg_midis('Shape of You'))
        pm = pretty_midi.PrettyMIDI(eg_songs('Merry Go Round of Life'))

        # pr = pm.get_piano_roll(100)
        # ic(pr.shape, pr.dtype, pr[75:80, 920:960])
        # # ic(np.where(pr > 100))
        #
        # instr0 = pm.instruments[0]
        # instr1 = pm.instruments[1]
        # ic(instr0.get_piano_roll()[76, 920:960])
        # ic(instr1.get_piano_roll()[76, 920:960])

        pmu = PrettyMidiUtil()
        pmu.plot_piano_roll(pm, fqs=100)
        # pmu.plot_piano_roll(instr0)
    # check_piano_roll()

    def test_show_in_plot():
        data = [('Chopin', [(1810, 1849 - 1810)]),
                ('Schumanns', [(1810, 1856 - 1810), (1819, 1896 - 1819)]),
                ('Brahms', [(1833, 1897 - 1833)])]
        xTicks = [(1810, '1810'),
                  (1848, '1848'),
                  (1897, '1897')]
        ghb = graph.primitives.GraphHorizontalBar()
        ghb.title = 'Romantics live long and not so long'
        ghb.data = data
        ghb.setTicks('x', xTicks)

        ghb.doneAction = None  # The `show` action calls figure.show() which doesn't seem to work in Pycharm
        ghb.process()
        # ghb.callDoneAction()
        plt.show()
    # test_show_in_plot()

    def test_piano_roll():
        fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        ic(fnm)

        scr = m21.converter.parse(fnm)
        ic(scr.id)
        # Looks like this doesn't work **sometimes**?
        # r = scr.plot('pianoroll', figureSize=(16, 9), doneAction=None)
        # ic(r)

        part = scr.parts[0]
        # plt_ = graph.plot.HorizontalBarPitchSpaceOffset(part.measures(25, 90), doneAction=None, figsize=(16, 9))
        # plt_.run()
        # ic(len(plt_.data), plt_.data)
        # plt.tight_layout()
        # ic(plt.xlim(), plt.ylim())
        # ic(plt.xticks(), plt.yticks())
        # # plt.xlim([0, 501])
        # plt.show()

        m2u = Music21Util()
        # ms = part.measures(20, 100)
        # m2u.plot_piano_roll(ms, s=10, e=30)
        # m2u.plot_piano_roll(scr, s=10, e=15)
        m2u.plot_piano_roll(part, s=20, e=40)
        # ic(type(ms), vars(ms), dir(ms))
        # ic(ms.measures(26, 30))
        # ic(mes, nums)
        # for mes in ms.measures(0, None):
        #     ic(mes)
    # test_piano_roll()

    def check_show_title():
        from .util import eg_songs

        fnm = eg_songs('Merry Go Round of Life', fmt='MXL')
        ic(fnm)
        scr = m21.converter.parse(fnm)
        ic(scr)
        # ic(len(dir(scr)))
        # ic(vars_(scr, include_private=False))
        meta = scr.metadata
        # ic(meta, vars(meta), vars_(meta))
        ic(meta.title, meta.composer)
        part_ch2 = scr.parts[1]
        ic(part_ch2, part_ch2.partName, part_ch2.metadata)
        # ic(vars(part_ch2), vars_(part_ch2))
        ic(part_ch2.activeSite.metadata.title)
    # check_show_title()