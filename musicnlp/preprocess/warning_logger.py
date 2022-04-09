from fractions import Fraction
from collections import Counter

from music21.meter import TimeSignature

from musicnlp.util import *
from musicnlp.util.music_lib import Dur, TsTup


class WarnLog:
    """
    Keeps track of warnings in music extraction

    JSON-serializable
    """
    MultTempo, MultTimeSig = 'Multiple Tempos', 'Multiple Time Signatures'
    MissTempo = 'Missing Tempo'
    InvTupSz, TupNoteOvl = 'Invalid Tuplet Size', 'Tuplet Notes Overlap'
    # InvTupNt = 'Invalid Tuplet Notes'
    InvTupDur, InvTupDurSv = 'Invalid Tuplet Durations', 'Invalid Tuplet Durations, Severe'
    RestInTup = 'Rest in Tuplet'
    HighPchOvl, HighPchOvlTup = 'Higher Pitch Overlap', 'Higher Pitch Overlap with Triplet'
    LowPchMakeup, LowPchMakeupRmv = 'Lower Pitch Makeup', 'Lower Pitch Makeup Removed'
    IncTimeSig, UncomTimeSig = 'Inconsistent Time Signatures', 'Uncommon Time Signature'
    NoteNotQuant, TupNoteQuant = 'Notes Beyond Quantization', 'Tuplet Notes Quantizable'
    InvBarDur = 'Invalid Bar Notes Duration'
    BarNoteGap = 'Gap in extracted Bar Notes'
    ExcecTupNote = 'Excessive Tuplet Chord Notes'
    EmptyStrt, EmptyEnd = 'Beginning Empty Bars', 'Ending Empty Bars'
    TYPES = [  # Warning types, ordered by severity
        EmptyStrt, EmptyEnd,
        MultTempo, MultTimeSig,
        MissTempo,
        IncTimeSig, UncomTimeSig,
        HighPchOvl, HighPchOvlTup,
        LowPchMakeup, LowPchMakeupRmv,
        InvTupSz,
        InvTupDur, InvTupDurSv,
        # InvTupNt,
        RestInTup,
        ExcecTupNote,
        TupNoteQuant,
        TupNoteOvl,
        NoteNotQuant,
        InvBarDur,
        BarNoteGap
    ]

    def __init__(self, name=f'Music Extraction Warn Log', verbose=True):
        self.warnings: List[Dict] = []
        self.idx_track = None
        self.args_func = None
        self.verbose = verbose

        self.logger = get_logger(name=name)
        self.yellow = MyTheme.yellow

    @staticmethod
    def warn_nm2str_tpl(warn_nm) -> str:
        """
        Map Warning to string output formatting template
        """
        if warn_nm == WarnLog.MultTempo:
            msg = '{warn_name}: More than one tempo observed - tempos: {tempos}'
        elif warn_nm == WarnLog.MultTimeSig:
            msg = '{warn_name}: More than one time signature observed - time_sigs: {time_sigs}'
        elif warn_nm == WarnLog.MissTempo:
            msg = '{warn_name}: No tempo found at 1st bar'
        elif warn_nm in [WarnLog.InvTupDur, WarnLog.InvTupDurSv]:
            msg = '{warn_name}: Tuplet durations don\'t sum up to 8th notes ' \
                  'at bar#{bar_num}, with offsets: {offsets}, durations: {durations} ' \
                  '- note durations distributed, and cropped to bar length if necessary'
        elif warn_nm == WarnLog.InvTupSz:
            msg = '{warn_name}: Tuplet with invalid number of notes added ' \
                  'at bar#{bar_num} - expect {n_expect}, got {n_got}'
        # elif nm == WarnLog.InvTupNt:
        #     msg = '{warn_name}: Tuplet with overlapping notes added at bar#{bar_num}, ' \
        #           'with offsets: {offsets}, durations: {durs} - notes are cut off'
        elif warn_nm == WarnLog.RestInTup:
            msg = '{warn_name}: Rest Notes observed in tuplet group at bar#{bar_num}' \
                  ' - total note {n_note}, rest count {n_rest}'
        elif warn_nm == WarnLog.ExcecTupNote:
            msg = '{warn_name}: Too much Chord notes in tuplet group at bar#{bar_num}' \
                  ' - note choices {note_choices}'
        elif warn_nm == WarnLog.UncomTimeSig:
            msg = '{warn_name}: Time Signature is uncommon' \
                  ' - Expect one of: {time_sig_expect}, got {time_sig_got}'
        elif warn_nm == WarnLog.IncTimeSig:
            msg = '{warn_name}: ratio of mode time signature below {threshold}' \
                  ' - #mode {n_bar_mode}, #total {n_bar_total}'
        elif warn_nm == WarnLog.HighPchOvlTup:
            msg = '{warn_name}: Higher pitch observed at bar#{bar_num}' \
                  ' - triplet truncated'
        elif warn_nm == WarnLog.HighPchOvl:
            msg = '{warn_name}: Later overlapping note with higher pitch observed ' \
                  'at bar#{bar_num} - prior note truncated'
        elif warn_nm == WarnLog.LowPchMakeup:
            msg = '{warn_name}: Later overlapping note with lower pitch but longer ending time observed ' \
                  'at bar#{bar_num} - truncated note appended'
        elif warn_nm == WarnLog.LowPchMakeupRmv:
            msg = '{warn_name}: Previously-added, truncated, lower-pitch makeup note overridden ' \
                  'by higher pitch note at bar#{bar_num} - makeup note removed'
        elif warn_nm == WarnLog.InvBarDur:
            msg = '{warn_name}: Note duration don\'t add up to bar max duration at bar#{bar_num}'
        elif warn_nm == WarnLog.BarNoteGap:
            msg = '{warn_name}: Slots with no extracted note found in bar#{bar_num}: ' \
                  'time_sig: {time_sig}, precision {precision}, unfilled ranges {unfilled_ranges}'
        elif warn_nm == WarnLog.TupNoteOvl:
            msg = '{warn_name}: Notes inside tuplet group are overlapping at bar#{bar_num} ' \
                  '- Note durations will be equally distributed'
        elif warn_nm == WarnLog.NoteNotQuant:
            msg = '{warn_name}: Note durations smaller than quantized slot at bar#{bar_num} ' \
                  '- Note durations approximated'
        elif warn_nm == WarnLog.EmptyStrt:
            msg = '{warn_name}: Empty bars observed at start of song in range: {bar_range}'
        elif warn_nm == WarnLog.EmptyEnd:
            msg = '{warn_name}: Empty bars observed at end of song in range: {bar_range}'
        else:
            assert warn_nm == WarnLog.TupNoteQuant
            msg = '{warn_name}: Tuplet notes of equal duration is quantizable at bar#{bar_num}' \
                  f' - Tuplet notes converted to normal notes'
        return msg

    def update(self, warn_: Dict):
        """
        :param warn_: Dictionary object specifying warning information
            nm: Warning name
            args - Dict: Metadata for the warning
            id: Warning entry id
            timestamp: Logging timestamp
        """
        assert 'warn_name' in warn_
        nm, args = warn_['warn_name'], warn_

        assert nm in WarnLog.TYPES
        if nm == WarnLog.MultTimeSig:
            assert 'time_sigs' in args
        elif nm == WarnLog.MultTempo:
            assert 'tempos' in args
        elif nm == WarnLog.MissTempo:
            pass
        elif nm == WarnLog.InvTupSz:
            assert all(k in args for k in ['bar_num', 'n_expect', 'n_got'])
        elif nm in [
            # WarnLog.InvTupNt,
            WarnLog.InvTupDur, WarnLog.InvTupDurSv,
            WarnLog.NoteNotQuant, WarnLog.TupNoteQuant, WarnLog.TupNoteOvl,
            WarnLog.InvBarDur
        ]:
            assert all(k in args for k in ['bar_num', 'offsets', 'durations'])
            if nm == WarnLog.InvBarDur:
                assert 'time_sig' in args
        elif nm == WarnLog.BarNoteGap:
            assert all(k in args for k in ['bar_num', 'time_sig', 'precision', 'unfilled_ranges'])
        elif nm == WarnLog.RestInTup:
            assert all(k in args for k in ['bar_num', 'n_rest', 'n_note'])

        elif nm in [WarnLog.HighPchOvl, WarnLog.HighPchOvlTup, WarnLog.LowPchMakeup, WarnLog.LowPchMakeupRmv]:
            assert 'bar_num' in args
        elif nm == WarnLog.UncomTimeSig:
            assert 'time_sig_expect' in args and 'time_sig_got' in args
        elif nm in [WarnLog.EmptyStrt, WarnLog.EmptyEnd]:
            assert 'bar_range' in args
        elif nm == WarnLog.ExcecTupNote:
            assert 'note_choices' in args
        else:
            assert nm == WarnLog.IncTimeSig
            assert all(k in args for k in ['time_sig', 'threshold', 'n_bar_total', 'n_bar_mode'])
        if self.args_func is not None:
            warn_ = self.args_func() | warn_

        if self.verbose:
            warn_out = {k: logi(v)+self.yellow for k, v in warn_.items()}
            warn_out['warn_name'] = sty.ef.bold + self.yellow + warn_out['warn_name']
            self.logger.warning((WarnLog.warn_nm2str_tpl(nm) + MyFormatter.RESET).format(**warn_out))
        self.warnings.append(warn_)

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.warnings)  # TODO: change column names?
        return df

    def start_tracking(self, args_func: Callable[[], Dict] = None):
        """
        Start tracking warnings added after the call

        :args_func: A function that returns a dictionary of metadata, merged to each `update` call
        """
        self.idx_track = len(self.warnings)
        self.args_func = args_func

    def end_tracking(self):
        """
        Removes prior internal warnings
        """
        self.warnings = []
        self.idx_track = len(self.warnings)

    @staticmethod
    def serialize_time_sig(ts: Union[TimeSignature, TsTup]):
        if isinstance(ts, tuple):
            return ts
        else:
            return ts.numerator, ts.denominator

    @staticmethod
    def serialize_durs(lst: List[Dur]):
        # e.g. `Fraction(1, 3)` => `(1, 3)`
        return [(e.numerator, e.denominator) if isinstance(e, Fraction) else e for e in lst]

    @staticmethod
    def serialize_warning(d: Dict):
        if 'offsets' in d:
            d['offsets'] = WarnLog.serialize_durs(d['offsets'])
        if 'durations' in d:
            d['durations'] = WarnLog.serialize_durs(d['durations'])
        if 'time_sig' in d:
            d['time_sig'] = WarnLog.serialize_time_sig(d['time_sig'])
        if 'time_sig_expect' in d:
            d['time_sig_expect'] = [WarnLog.serialize_time_sig(ts) for ts in d['time_sig_expect']]
        if 'time_sig_got' in d:
            d['time_sig_got'] = WarnLog.serialize_time_sig(d['time_sig_got'])
        return d

    def tracked(self, exp: str = 'summary') -> Union[Dict, List[Dict]]:
        """
        Statistics of warnings since tracking started

        :param exp: Export mode, one of [`summary`, `raw`, `serialize`]
        """

        if exp == 'summary':
            return Counter(w['warn_name'] for w in self.warnings[self.idx_track:])
            # return ', '.join((f'{logi(k)}: {logi(v)}' for k, v in counts.items()))
        elif exp == 'serialize':
            return [WarnLog.serialize_warning(wn) for wn in self.warnings[self.idx_track:]]
        else:
            return self.warnings[self.idx_track:]
