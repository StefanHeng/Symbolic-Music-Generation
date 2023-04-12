import os
import re
import math
import json
import random
import pickle
from os.path import join as os_join
from copy import deepcopy
from typing import List, Tuple, Dict, Callable, Union, Optional, Any
from fractions import Fraction
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from stefutil import *
from musicnlp.util import *
from musicnlp.util.music_lib import Dur
from musicnlp.vocab import (
    COMMON_TEMPOS, COMMON_TIME_SIGS, get_common_time_sig_duration_bound,
    MusicVocabulary, MusicTokenizer, key_str2enum, ElmType
)
from musicnlp.preprocess import WarnLog, transform, MusicConverter
from musicnlp.trainer import load_wordpiece_tokenizer, load_pairmerge_tokenizer
from musicnlp.postprocess.music_stats import MusicStats


@dataclass
class PlotOutputPair:
    df: pd.DataFrame = None
    ax: plt.Axes = None
    meta: Dict[str, Any] = None


class MusicVisualize:
    """
    Visualize dataset info given json as extracted input representation

    See `preprocess.music_export.py`
    """
    key_dnm = 'dataset_name'
    color_rare = hex2rgb('#E06C75', normalize=True)
    pattern_frac = re.compile(r'^(?P<numer>\d+)/(?P<denom>\d+)$')

    rare_token_types = ['time_sig', 'tempo', 'pitch', 'duration']

    def __init__(
            self, filename: Union[str, List[str]], dataset_name: Union[str, List[str]] = None,
            color_palette: str = 'husl', hue_by_dataset: bool = True, cache: str = None,
            subset: Union[float, bool] = None, subset_bound: int = 4096, pitch_kind: str = 'degree'
    ):
        """
        :param filename: Path to a json dataset, or a list of paths, in which case datasets are concatenated
            See `preprocess.music_export.py`
        :param dataset_name: Datasets names, if given, should correspond to filenames
        :param hue_by_dataset: If true, automatically color-code statistics by dataset name
        :param subset: If given, a subset of songs will be taken for plotting for
            datasets with #song above `subset_bound`
        """
        self.dset: Dict
        self._prec, self.tokenizer, self.wp_tokenizer, self.pm_tokenizer = None, None, None, None
        self.vocab, self.states, self.mc = None, None, None
        self.sr, self.sr_stat, self.ak = None, None, None
        self.pitch_kind = pitch_kind

        self._df = None
        self.cache = cache
        self.logger = get_logger('Music Visualizer')
        d_log = dict(cache=cache, subset=subset, subset_bound=subset_bound)
        self.logger.info(f'Initializing {pl.i(self.__class__.__qualname__)} with {pl.i(d_log)}... ')
        self.logger.info('Getting global stats... ')
        if cache:
            fnm = f'{self.cache}.pkl'
            path = os_join(u.plot_path, 'cache')
            os.makedirs(path, exist_ok=True)
            path = os_join(path, fnm)
            if os.path.exists(path):
                self.logger.info(f'Loading cached stats from {pl.i(path)}... ')
                with open(path, 'rb') as f:
                    d = pickle.load(f)
                    self.dset, self._df = d['dset'], d['df']
                    self._set_meta()
            else:
                self.dset = self._get_dset(filename, dataset_name, subset=subset, subset_bound=subset_bound)
                self._set_meta()
                self._df = self._get_song_info()
                with open(path, 'wb') as f:
                    pickle.dump(dict(dset=self.dset, df=self._df), f)
                self.logger.info(f'Cached stats saved to {pl.i(path)} ')
        else:
            self.dset = self._get_dset(filename, dataset_name)
            self._set_meta()

        self.color_palette = color_palette
        if hue_by_dataset:
            assert dataset_name is not None, f'{pl.i("dataset_name")} is required for color coding'
        self.hue_by_dataset = hue_by_dataset
        self.dnms = dataset_name

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._get_song_info()
        return self._df

    def _get_dset(self, filename, dataset_name, subset=None, subset_bound=None):
        def _load_single(f_: str, dnm: str = None) -> Dict:
            self.logger.info(f'Loading JSON dataset {pl.i(stem(f_))}... ')
            with open(f_, 'r') as f:
                ds = json.load(f)
            if dnm:
                songs = ds['music']
                if subset_bound and len(songs) > subset_bound:
                    # random.seed(77)  # TODO: remove after fixed error
                    songs = random.sample(songs, round(len(songs) * subset))
                for s in songs:
                    s['dataset_name'] = dnm
                ds['music'] = songs
            return ds
        if isinstance(filename, str):
            if dataset_name:
                assert isinstance(dataset_name, str), \
                    f'Dataset name given should be a string for single filename, ' \
                    f'but got {pl.i(dataset_name)} with type {pl.i(type(dataset_name))}'
            return _load_single(filename, dataset_name)
        else:
            if dataset_name:
                assert isinstance(dataset_name, list), \
                    f'Dataset name given should be a list for multiple filenames, ' \
                    f'but got {pl.i(dataset_name)} with type {pl.i(type(dataset_name))}'
            else:
                dataset_name = [None] * len(filename)
            dset = [_load_single(f, dnm) for f, dnm in zip(filename, dataset_name)]
            assert all(ds['extractor_meta'] == dset[0]['extractor_meta'] for ds in dset)
            return dict(
                music=sum([d['music'] for d in dset], []),
                extractor_meta=dset[0]['extractor_meta']
            )

    @property
    def prec(self) -> int:
        if not self._prec:
            self._prec = get(self.dset, 'extractor_meta.precision')
            assert self.prec >= 2
        return self._prec

    @property
    def n_song(self) -> int:
        return len(self.dset['music'])

    def _set_meta(self):
        self.tokenizer = MusicTokenizer(precision=self.prec, pitch_kind=self.pitch_kind)
        self.wp_tokenizer = load_wordpiece_tokenizer(pitch_kind=self.pitch_kind, precision=self.prec)
        self.pm_tokenizer = load_pairmerge_tokenizer(pitch_kind=self.pitch_kind, precision=self.prec)

        assert self.wp_tokenizer.precision == self.prec
        self.vocab: MusicVocabulary = self.tokenizer.vocab
        if self.pitch_kind == 'step':
            self.vocab_sp = self.vocab
        else:
            self.vocab_sp = MusicVocabulary(pitch_kind='step', precision=self.prec)
        self.stats = MusicStats(prec=self.prec, pitch_kind=self.pitch_kind)
        self.converter = self.stats.converter
        self.mc: MusicConverter = self.stats.converter
        sr_vocab = self.vocab if self.pitch_kind == 'step' else MusicVocabulary(pitch_kind='step', precision=self.prec)
        self.sr = transform.SanitizeRare(vocab=sr_vocab)
        # Another sanitize rare that preserves the original rare time sig & tempo
        self.sr_stat = transform.SanitizeRare(vocab=sr_vocab, rare_pitch_only=True)
        self.ak = transform.AugmentKey(vocab=self.vocab)

    def _extract_song_info(self, d: Dict, it=None) -> Dict:
        d = deepcopy(d)
        title = d['title']
        scr = d.pop('score')

        scr_ = self.sr(scr)
        keys = d['keys']
        # Needed for tokenization, just pick the most-confident key for simplicity
        scr_ = self.ak((scr_, max(keys, key=keys.get)))

        toks = self.tokenizer.tokenize(scr_)
        wp_toks = self.wp_tokenizer.tokenize(scr_, mode='char')  # just need len, efficient
        pm_toks = self.pm_tokenizer.tokenize(scr_)
        d['n_token'], d['n_wp_token'], d['n_pm_token'] = len(toks), len(wp_toks), len(pm_toks)

        # Count #bars with no melody and no bass
        out = self.converter.str2music_elms(scr_, pitch_kind=self.pitch_kind)
        lst_elms = out.elms_by_bar
        n_empty_channel = dict(melody=0, bass=0)
        for i, elms in enumerate(lst_elms):
            notes = self.converter.split_notes(elms)
            n_non_rest_notes = dict(melody=0, bass=0)
            for channel in n_non_rest_notes.keys():
                for n in notes[channel]:
                    if n.type == ElmType.tuplets:
                        n_non_rest_notes[channel] += 1  # Assume tuplets must contain at least one rest note
                    elif n.type == ElmType.note:
                        if n.meta[0] != self.vocab.rest_pitch_meta:  # pitch meta is not rest
                            n_non_rest_notes[channel] += 1
            for channel, n_non in n_non_rest_notes.items():
                if n_non == 0:  # all notes are rest notes
                    n_empty_channel[channel] += 1
            if n_non_rest_notes['melody'] == 0:
                if n_non_rest_notes['bass'] != 0:
                    # Should be rare, see `MusicExtractor::extract_notes`
                    self.logger.warning(f'Bass has non-rest notes when melody is empty at bar {pl.i(i)} '
                                        f'with {pl.i(notes)}: Piece {pl.i(title)} is likely low in quality ')
        d['n_empty_channel_by_bar'] = n_empty_channel
        d['n_empty_channel_by_song'] = {k: 1 if v > 0 else 0 for k, v in n_empty_channel.items()}

        # number of pitch token
        d['n_tuplet_note'] = dict(Counter([len(e.meta[0]) for e in out.elms if e.type == ElmType.tuplets]))
        d['tuplet_duration_count'] = dict(Counter(e.meta[1] for e in out.elms if e.type == ElmType.tuplets))

        # Get count of uncommon token and total token by type
        type2total_count = {t: 0 for t in MusicVisualize.rare_token_types}
        type2rare_count = {t: 0 for t in MusicVisualize.rare_token_types}
        raw_toks = self.tokenizer.tokenize(scr)
        for tok in raw_toks:  # Pitch kind is step, not sanitized, the raw extraction output
            typ = self.vocab.type(tok).name
            is_rare = self.vocab_sp.is_rare_token(tok)
            if typ in MusicVisualize.rare_token_types:
                if is_rare:
                    type2rare_count[typ] += 1
                type2total_count[typ] += 1
        d['total_token_count'] = type2total_count
        d['rare_token_count'] = type2rare_count
        d['raw_count'] = dict(Counter(raw_toks))

        if it:
            d_log = dict(
                n_token=pl.i(d['n_token']), n_wp_token=pl.i(d['n_wp_token']), n_pm_token=pl.i(len(pm_toks)),
                title=pl.i(title)
            )
            it.set_postfix(d_log)
        counter_toks = Counter(toks)
        d['n_bar'] = counter_toks[self.vocab.start_of_bar]
        d['n_tup'] = counter_toks[self.vocab.start_of_tuplet]
        del d['warnings']

        scr_stat = self.sr_stat(scr)
        # Don't need to insert proper key cos key stats already considered, but need to shift pitch
        scr_stat = self.ak((scr_stat, max(keys, key=keys.get)))
        toks_stat = self.tokenizer.tokenize(scr_stat)
        ttc = self.stats.vocab_type_counts(toks_stat)
        # Only 1 per song
        d['tempo'] = list(ttc['tempo'].keys())[0]
        d['time_sig'] = (numer, denom) = list(ttc['time_sig'].keys())[0]
        d['time_sig_str'] = f'{numer}/{denom}'
        d['keys_unweighted'] = {k: 1 for k in keys}  # to fit into `_count_by_dataset`

        d['duration_count'], d['pitch_count'] = ttc['duration'], ttc['pitch']
        d['weighted_pitch_count'] = self.stats.weighted_pitch_counts(toks_stat)
        return d

    def _get_song_info(self):
        entries: List[Dict] = self.dset['music']

        # concurrent = True
        concurrent = False
        if concurrent:
            tqdm_args = dict(desc='Extracting song info', unit='song', chunksize=128)
            ds = conc_map(self._extract_song_info, entries, with_tqdm=tqdm_args, mode='process', n_worker=6)
        else:
            it = tqdm(entries, desc='Extracting song info', unit='song')
            ds = []
            for song in it:
                ds.append(self._extract_song_info(song, it))
        return pd.DataFrame(ds)

    def hist_wrapper(
            self, data: pd.DataFrame = None, col_name: str = None,
            title: str = None, xlabel: str = None, ylabel: str = None, yscale: str = None,
            callback: Callable = None, show: bool = True, save: bool = False,
            upper_percentile: float = None, show_title: bool = True, **kwargs
    ):
        self.logger.info('Plotting... ')
        args = dict(palette=self.color_palette, kde=True, kde_kws=dict(gridsize=2048 * 3), common_norm=False) | kwargs
        if self.hue_by_dataset:
            args['hue'] = 'dataset_name'
        data = data if data is not None else self.df
        ax = sns.histplot(data=data, x=col_name, **args)
        if self.hue_by_dataset:
            legend_title = ' '.join([s.capitalize() for s in self.key_dnm.split('_')])
            plt.gca().get_legend().set_title(legend_title)
        plt.xlabel(xlabel)
        ylab_default = 'count'
        if show_title and title is not None:
            plt.title(title)
        if upper_percentile:
            vs = data[col_name]
            q = upper_percentile if isinstance(upper_percentile, (float, int)) else 99.7  # ~3std

            def get_range(vals: np.ndarray):
                return vals.min(), np.percentile(vals, q=q)
            if self.hue_by_dataset:
                mis, mas = [], []
                for dnm, d in data.groupby('dataset_name'):
                    mi, ma = get_range(d[col_name])
                    mis.append(mi)
                    mas.append(ma)
                mi, ma = min(mis), max(mas)
            else:
                mi, ma = get_range(vs)
            ax.set_xlim([mi, ma])
        stat = args.get('stat', None)
        if upper_percentile or stat in ['density', 'percent']:
            ylab_default = stat
            if stat == 'percent':
                ylab_default = f'{ylab_default} (%)'
            assert (not yscale) or yscale == 'linear'
        plt.ylabel(ylabel or ylab_default)
        if yscale:
            plt.yscale(yscale)
        if callback is not None:
            callback(ax)
        if save:
            title = title.replace('w/', 'with')
            save_fig(title)
        elif show:
            plt.show()
        return ax

    def token_length_dist(self, tokenize_scheme: str = 'vanilla', **kwargs):
        col_nm, title = 'n_token', 'Distribution of token length'
        if tokenize_scheme == 'wordpiece':
            col_nm = 'n_wp_token'
            title = f'{title} w/ WordPiece tokenization'
        elif tokenize_scheme == 'pairmerge':
            col_nm = 'n_pm_token'
            title = f'{title} w/ PairMerge tokenization'
        args = dict(col_name=col_nm, title=title, xlabel='#token')
        if kwargs is not None:
            args.update(kwargs)
        return self.hist_wrapper(**args)

    def bar_count_dist(self, **kwargs):
        return self.hist_wrapper(col_name='n_bar', title='Distribution of #bars', xlabel='#bar', **kwargs)

    def tuplet_count_dist(self, **kwargs) -> PlotOutputPair:
        title = 'Distribution of #tuplets'
        args = dict(col_name='n_tup', title=title, xlabel='#tuplet', upper_percentile=True) | (kwargs or dict())
        df = self.df.groupby([self.key_dnm, 'n_tup']).size().reset_index(name='count')
        return PlotOutputPair(df=df, ax=self.hist_wrapper(**args))

    def tuplet_n_note_dist(self, **kwargs) -> PlotOutputPair:
        self.logger.info('Getting stats... ')
        k = 'n_tuplet_note'
        df = self._count_by_dataset(k)

        def callback(ax):
            import matplotlib.ticker as plticker

            loc = plticker.MultipleLocator(base=2)  # Use more frequent ticks cos the actual # matters
            ax.xaxis.set_major_locator(loc)
            ma, mi = df[k].max(), df[k].min()
            ax.set_xlim([mi, ma])
        title = 'Distribution of #notes in tuplets'
        ax_ = self.hist_wrapper(
            data=df, col_name='n_tuplet_note', weights='count', title=title, xlabel='#note in tuplets',
            discrete=True, kde=False, callback=callback, **kwargs
        )
        return PlotOutputPair(df=df, ax=ax_)

    def song_duration_dist(self, **kwargs):
        def callback(ax):
            x_tick_vals = [v for v in plt.xticks()[0] if v >= 0]
            ax.set_xticks(x_tick_vals, labels=[sec2mmss(v) for v in x_tick_vals])
        title = 'Distribution of song duration'
        args = dict(col_name='duration', title=title, xlabel='duration (mm:ss)', callback=callback)
        if kwargs is not None:
            args.update(kwargs)
        return self.hist_wrapper(**args)

    @staticmethod
    def _cat_plot_force_render_n_reduce_lim(ax):
        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks)  # Hack to force rendering for `show`, TODO: better ways?
        ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()])

        mi, ma = min(x_ticks), max(x_ticks)
        ax.set_xlim([mi - 0.5, ma + 0.5])  # Reduce the white space on both sides

    def time_sig_dist(self, kind: str = 'hist', **kwargs) -> PlotOutputPair:
        self.logger.info('Getting stats... ')

        def callback(ax):
            plt.gcf().canvas.draw()  # so that labels are rendered
            xtick_lbs = ax.get_xticklabels()
            com_tss = [f'{ts[0]}/{ts[1]}' for ts in COMMON_TIME_SIGS]
            for t in xtick_lbs:
                txt = t.get_text()
                if '/' in txt:
                    numer, denom = MusicVisualize._parse_frac(txt)
                    t.set_usetex(True)
                    t.set_text(MusicVisualize._frac2tex_frac(numer, denom, enforce_denom=False))
                else:
                    t.set_text(txt)
                if txt not in com_tss:
                    t.set_color(self.color_rare)
            MusicVisualize._cat_plot_force_render_n_reduce_lim(ax)
        ca(dist_plot_type=kind)
        title = 'Distribution of Time Signature'
        if kind == 'hist':
            tss_rare = [ts for ts in self.df.time_sig.unique() if ts not in COMMON_TIME_SIGS and ts != (None, None)]
            tss_rare = sorted(tss_rare, key=lambda ts: ts[0]/ts[1])  # sort by duration of the time signature
            tss = COMMON_TIME_SIGS + tss_rare
            tss_print = [f'{numer}/{denom}' for numer, denom in tss]
            df_col2cat_col(self.df, 'time_sig_str', tss_print)
            c_nm, xlab = 'time_sig_str', 'Time Signature'
            args = dict(col_name=c_nm, title=title, xlabel=xlab, yscale='log', kde=False, callback=callback)
            args.update(kwargs)
            df = self.df.groupby([self.key_dnm, c_nm]).size().reset_index(name='count')
            return PlotOutputPair(df=df, ax=self.hist_wrapper(**args))
        else:
            df = self._count_by_dataset('time_sig')
            df['time_sig'] = df['time_sig'].map(lambda ts: f'{ts[0]}/{ts[1]}')  # so that the category ordering works
            args = dict(
                data=df, x='time_sig', y='count',
                hue=self.key_dnm, x_order=self.df.time_sig_str.cat.categories,
                xlabel='Time Signature', ylabel='count', title=title, yscale='log', width=False, callback=callback
            )
            args |= kwargs
            return PlotOutputPair(df=df, ax=barplot(**args))

    def tempo_dist(self, **kwargs):
        def callback(ax):
            mi, ma = ax.get_xlim()
            ax.set_xlim([max(mi, 0), ma])  # cap to positive tempos
            plt.gcf().canvas.draw()  # so that labels are rendered
            xtick_lbs = ax.get_xticklabels()
            for t in xtick_lbs:
                # matplotlib encoding for negative int label
                if int(re.sub(u'\u2212', '-', t.get_text())) not in COMMON_TEMPOS:
                    t.set_color(self.color_rare)
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels([t.get_text() for t in xtick_lbs])  # Hack
        title, xlab = 'Distribution of Tempo', 'Tempo (bpm)'
        args = dict(col_name='tempo', title=title, xlabel=xlab, kde=False, callback=callback) | kwargs
        # get tempo occurrence counts by dataset
        df = self.df.groupby([self.key_dnm, 'tempo']).size().reset_index(name='count')
        return PlotOutputPair(df=df, ax=self.hist_wrapper(**args))

    def key_dist(self, weighted=True, **kwargs):
        self.logger.info('Getting stats... ')
        key_pattern = re.compile(r'^(?P<key>[A-G])(?P<shift>[#b])?(?P<class>.*)$')
        cls2cls_post = dict(Major='maj', Minor='mi')

        def key2shorter_key(key: str) -> str:
            m = key_pattern.match(key)
            assert m is not None
            k_, sh, cls = m.group('key'), m.group('shift'), m.group('class')
            if sh:
                k_ = f'{k_}{sh}'
            return f'{k_}{cls2cls_post[cls]}'
        keys = list(key_str2enum.keys())
        key2shorter_key = {k: key2shorter_key(k) for k in keys}

        k = 'keys' if weighted else 'keys_unweighted'
        df = self._count_by_dataset(k)
        df.rename(columns={k: 'key'}, inplace=True)
        df_col2cat_col(df, 'key', categories=keys)
        df.key = df.key.apply(lambda k: key2shorter_key[k])

        title, xlab = 'Distribution of Key', 'Key'
        if weighted:
            title = f'{title}, weighted by confidence'
        args = dict(data=df, col_name='key', weights='count', kde_kws=dict(bw_adjust=0.25), title=title, xlabel=xlab)
        args |= kwargs
        return self.hist_wrapper(**args)

    def _count_column(self, col_name: str) -> Dict[str, Counter]:
        dnm2counts = defaultdict(Counter)
        for dnm in self.df[self.key_dnm].unique():
            df = self.df[self.df[self.key_dnm] == dnm]
            if isinstance(next(iter(df[col_name])), dict):
                for d in df[col_name]:
                    dnm2counts[dnm].update(d)
            else:  # TODO: didn't check
                dnm2counts[dnm].update(df[col_name].value_counts())
        return dnm2counts

    def _count_by_dataset(self, col_name: str) -> pd.DataFrame:
        dfs, cols = [], [col_name, 'count', self.key_dnm]
        for dnm, counts in self._count_column(col_name).items():
            dfs.append(pd.DataFrame([(k, v, dnm) for k, v in counts.items()], columns=cols))
        return pd.concat(dfs, ignore_index=True)

    def note_pitch_dist(self, weighted=True, **kwargs):
        self.logger.info('Getting stats... ')
        k = 'weighted_pitch_count' if weighted else 'pitch_count'
        df = self._count_by_dataset(k)
        df.rename(columns={k: 'pitch'}, inplace=True)
        ma, mi = df.pitch.max(), df.pitch.min()
        mr = self.vocab.tok2meta(self.vocab.rest)
        if self.pitch_kind != 'midi':
            mr = mr[0]
        assert mi == mr  # sanity check

        def callback(ax):
            plt.gcf().canvas.draw()
            pch_ints = [-1, *range(6, ma + 6, 6)]
            ax.set_xticks(pch_ints, labels=[self.tokenizer.vocab.pitch_midi2name(p) for p in pch_ints])
        title, xlab = 'Distribution of Pitch', 'Pitch'
        if weighted:
            title = f'{title}, weighted by duration'
        return self.hist_wrapper(
            data=df, col_name='pitch', weights='count', discrete=True,
            title=title, xlabel=xlab, kde=True, callback=callback, **kwargs
        )

    @staticmethod
    def _parse_frac(s: str) -> Tuple[int, int]:
        m = MusicVisualize.pattern_frac.match(s)
        assert m
        return int(m.group('numer')), int(m.group('denom'))

    @staticmethod
    def _frac2tex_frac(numer, denom, enforce_denom: bool = True) -> str:
        if enforce_denom:
            assert denom != 1
        return rf'$\nicefrac{{{numer}}}{{{denom}}}$'

    def note_duration_dist(self, kind='hist', note_type: str = 'all', **kwargs) -> Optional[PlotOutputPair]:
        """
        Distribution of note durations

        .. note:: Tuplet notes contribute to a single duration, i.e. all quantized durations
        """
        self.logger.info('Getting stats... ')
        ca(dist_plot_type=kind)
        ca.check_mismatch('Plot Note Type', note_type, ['all', 'tuplet'])
        is_tup = note_type == 'tuplet'
        nt = 'Tuplet Note' if is_tup else 'Note'
        title, xlab = f'Distribution of {nt} Duration', 'Duration (quarter length)'
        if kind == 'hist':
            k = 'tuplet_duration_count' if is_tup else 'duration_count'
            df = self._count_by_dataset(k)
            df.rename(columns={k: 'duration'}, inplace=True)
            d_uniq = df.duration.unique()
            d_uniq = [d for d in d_uniq if d is not None]
            bound = min(max(d_uniq), get_common_time_sig_duration_bound())

            df.duration = df.duration.apply(str)
            df_col2cat_col(df, 'duration', [str(d) for d in sorted(d_uniq)])

            def callback(ax):
                plt.gcf().canvas.draw()
                xtick_lbs = ax.get_xticklabels()

                for t in xtick_lbs:
                    txt = t.get_text()
                    if '/' in txt:
                        numer, denom = MusicVisualize._parse_frac(txt)
                        val = numer / denom
                        t.set_usetex(True)
                        t.set_text(MusicVisualize._frac2tex_frac(numer, denom))
                    else:
                        val = int(txt)
                        t.set_text(txt)
                    if val > bound:
                        t.set_color(self.color_rare)
                MusicVisualize._cat_plot_force_render_n_reduce_lim(ax)
            ax_ = self.hist_wrapper(
                data=df, col_name='duration', weights='count', discrete=True, kde=False,
                title=title, xlabel=xlab,
                callback=callback,
                **kwargs
            )
            return PlotOutputPair(df=df, ax=ax_)
        else:  # TODO: doesn't look good
            counts = Counter()
            for d in self.df.duration_count:
                counts.update(d)
            bound = min(max(counts.keys()), get_common_time_sig_duration_bound())
            assert bound.is_integer()
            bound = int(bound)

            # Number of colors needed for an integer group
            # e.g. precision = 5, smallest duration 1/8, needs 4 colors, for [1, 1/2, 1/4, 1/8]
            n_color_per_group = (self.prec-2) + 1
            n_color = n_color_per_group * bound

            durs_above = sorted([d for d in counts.keys() if d > bound])
            durs: List[Dur] = self.vocab.get_durations(bound=bound, exp='dur')
            durs_str = durs + durs_above
            cs = sns.color_palette(palette='husl', n_colors=n_color + len(durs_above))

            # Colors are locally ordered by precision, added with shades based on global magnitude
            def get_category_idx(c: Union[int, Fraction]):
                if isinstance(c, int):
                    idx_local, group = 0, c-1
                else:
                    idx_local = math.log2(c.denominator)
                    assert idx_local.is_integer()
                    idx_local, group = int(idx_local), math.floor(c)
                return idx_local + n_color_per_group * group
            cs = [cs[get_category_idx(c)] for c in durs] + [cs[-1] for _ in durs_above]
            y = [counts[d] for d in durs_str]
            return barplot(x=durs_str, y=y, palette=cs, xlabel=xlab, ylabel='count', yscale='log', title=title)

    def token_coverage_dist(self, ratio: float = 0.95) -> PlotOutputPair:
        """
        Plots side-by-side and cumulative distribution of tokens for each dataset
        """
        df = self._count_by_dataset('raw_count')
        df.rename(columns={'raw_count': 'token'}, inplace=True)
        # split into dataframes by dataset, convert each into dict
        dfs = {dnm: df[df[self.key_dnm] == dnm] for dnm in self.dnms}
        counts = {dnm: dict(zip(df.token, df['count'])) for dnm, df in dfs.items()}
        null_count = {t: 0 for t in self.vocab_sp.tok2id}  # for showing counts in stats vocab
        mic(len(null_count))
        fig, axs = plt.subplots(nrows=1, ncols=2)  # side-by-side, frequency plot and cumulative plot
        cs = sns.color_palette(palette='husl', n_colors=len(self.dnms))

        def plot_single(dnm: str = None, i: int = None):
            # tokens that don't appear also take a slot => ensures similar array lengths for all datasets in plot
            # lengths can still differ for uncommon tokens, e.g. rare time signature
            c = null_count | counts[dnm]
            # get list of token names, sorted in descending order of count,
            # also get the corresponding counts as numpy array
            tokens, cts = zip(*sorted(c.items(), key=lambda x: x[1], reverse=True))
            cts = np.array(cts)
            # get cumulative sum of counts, normalized by total count
            total = cts.sum()
            cts_cumsum = np.cumsum(cts) / total
            cts = cts / total

            # get subset of tokens that cover `coverage_ratio` of the total count
            # get index that covers standard deviations
            idx1 = np.argmax(cts_cumsum > 0.68)  # 1std
            idx2 = np.argmax(cts_cumsum > 0.95)  # 2std
            idx25 = np.argmax(cts_cumsum > 0.986)  # 2.5std
            idx3 = np.argmax(cts_cumsum > 0.997)  # 3std
            idx = np.argmax(cts_cumsum > ratio)
            toks = tokens[:idx]

            c_cover = Counter({t: c[t] for t in toks})

            # plot as a line
            c = cs[i]
            ln_args = LN_KWARGS | dict(c=c, lw=0.4)
            axs[0].plot(cts[:idx] * 100, label=dnm, **ln_args)
            axs[1].plot(cts_cumsum[:idx] * 100, label=dnm, **ln_args)

            args = dict(lw=0.25, color=c)
            vs = [
                # (idx1, 0.68, f'{dnm} 68% at vsz={idx1}'),
                (idx2, 0.95, f'{dnm} 95% at vsz={idx2}'),
                (idx25, 0.986, f'{dnm} 98.6% at vsz={idx25}'),
                (idx3, 0.997, f'{dnm} 99.7% at vsz={idx3}')
            ]

            def plot_vlns(ax):
                # plot horizontal lines at 68%, 95%, 98.6%, 99.7% coverage and coverage ratio up until plot ends
                for idx_, r, lb in vs:
                    if idx_ <= idx:
                        ax.axvline(x=idx_, **args, label=lb)
            plot_vlns(axs[0])
            plot_vlns(axs[1])
            return dict(n=len(toks), counter=c_cover)
        metas = {dnm: plot_single(dnm, i) for i, dnm in enumerate(self.dnms)}

        x_max = axs[1].get_xlim()[1]
        c_ln = sns.color_palette(palette='husl', n_colors=8)[-1]
        args_ = dict(xmin=0, xmax=x_max, lw=0.15, color=c_ln)
        axs[1].axhline(y=ratio * 100, **args_, label=f'{ratio * 100}% coverage')

        ma = axs[0].get_ylim()[1]  # Clip min y to 0
        axs[0].set_ylim(0, ma)
        axs[1].set_ylim(0, 100)

        fig.supxlabel('Vocabulary size')
        fig.supylabel('Dataset Token Coverage (%)')
        axs[0].set_title('Token Frequency')
        axs[1].set_title('Token Cumulative Distribution')
        axs[1].legend()
        fig.suptitle('Token Coverage')
        return PlotOutputPair(ax=axs, meta=metas)

    def empty_channel_ratio(self) -> PlotOutputPair:
        """
        Plots the ratio of empty melody and bass channels by measure and by song
        """
        df_b = self._count_by_dataset('n_empty_channel_by_bar')
        df_s = self._count_by_dataset('n_empty_channel_by_song')
        # divide by total number of bar and song respectively, for each dataset, to get ratio
        n_bar_by_dataset = self.df.groupby(self.key_dnm).n_bar.sum()
        n_song_by_dataset = self.df.groupby(self.key_dnm).size()
        # divide count by the corresponding dataset
        k_p = 'percent'
        df_b[k_p] = df_b.apply(lambda x: x['count'] / n_bar_by_dataset[x[self.key_dnm]], axis=1)
        df_s[k_p] = df_s.apply(lambda x: x['count'] / n_song_by_dataset[x[self.key_dnm]], axis=1)

        # rename melody & bass, bar & song into a single column to merge into single dataframe
        k = 'ratio'
        df_b.rename(columns={'n_empty_channel_by_bar': k}, inplace=True)
        df_s.rename(columns={'n_empty_channel_by_song': k}, inplace=True)
        df_b[k] = df_b[k].apply(lambda x: f'Empty {x} bars')
        df_s[k] = df_s[k].apply(lambda x: f'Song w/ Empty {x}')
        df = pd.concat([df_b, df_s]).reset_index(drop=True)
        df[k_p] *= 100

        title = 'Ratio of Empty Notes by Channels & by Measure/Song'
        ax = barplot(
            x=k, y=k_p, hue=self.key_dnm, data=df, xlabel='ratio kind', ylabel='percent (%)', title=title,
            hue_order=self.dnms
        )
        return PlotOutputPair(df=df, ax=ax)

    def tuplet_duration_ratio(self) -> PlotOutputPair:
        """
        Plots the ratio of tuplet durations, among all notes (tuplets and regular notes),
            by occurrence and by total duration
        """
        df_all = self._count_by_dataset('duration_count')
        df_tup = self._count_by_dataset('tuplet_duration_count')
        # multiply the duration count and the count column and sum to get total duration
        k_w, k_c = 'weighted', 'count'
        df_all[k_w] = df_all.apply(lambda x: x['duration_count'] * x['count'], axis=1)
        df_tup[k_w] = df_tup.apply(lambda x: x['tuplet_duration_count'] * x['count'], axis=1)

        def _get_ratio(key_dur: str) -> pd.DataFrame:
            # sum up total duration by dataset
            df_all_ = df_all[[self.key_dnm, key_dur]].groupby(self.key_dnm).sum()
            df_tup_ = df_tup[[self.key_dnm, key_dur]].groupby(self.key_dnm).sum()
            ret = (df_tup_ / df_all_).reset_index()  # divide by total duration to get a ratio for each dataset
            ret[key_dur] = ret[key_dur].astype(float)
            return ret
        weighted, occur = _get_ratio(key_dur=k_w), _get_ratio(key_dur=k_c)
        # unify duration keys, merge into single dataframe with corresponding ratio kind
        k_r, k_k = 'ratio', 'kind'
        weighted.rename(columns={k_w: k_r}, inplace=True)
        occur.rename(columns={k_c: k_r}, inplace=True)
        weighted[k_k], occur[k_k] = 'weighted', 'occurrence'
        df = pd.concat([weighted, occur]).reset_index(drop=True)
        df[k_r] *= 100

        def callback(ax):
            mi, ma = ax.get_ylim()
            ax.set_ylim([mi, max(ma, 10)])  # set y limit to 10 if it is smaller
        title, ylab = 'Ratio of Tuplet Durations by Occurrence & Weighted by Duration', 'percent (%)'
        ax_ = barplot(
            x=k_k, y=k_r, hue=self.key_dnm, data=df, xlabel='ratio type', ylabel=ylab, title=title, callback=callback,
            hue_order=self.dnms
        )
        return PlotOutputPair(df=df, ax=ax_)

    def rare_token_ratio(self) -> PlotOutputPair:
        """
        Plots the ratio of uncommon tokens by token type and by occurrence
        """
        # Get counts of rare and total tokens for each rare type
        df_total = self._count_by_dataset('total_token_count')
        df_rare = self._count_by_dataset('rare_token_count')
        # divide to get ratio for each type & each dataset
        df = df_total.copy()
        df.drop(columns=['count'], inplace=True)
        df.rename(columns=dict(total_token_count='kind'), inplace=True)
        df['ratio'] = (df_rare['count'] / df_total['count']) * 100

        title = 'Ratio of Uncommon Tokens by Token Type'
        ax = barplot(
            x='kind', y='ratio', hue=self.key_dnm, data=df, xlabel='token type', ylabel='percent (%)', title=title,
            hue_order=self.dnms
        )
        return PlotOutputPair(df=df, ax=ax)

    def warn_info(self, per_dataset: bool = False, as_counts=True) -> pd.DataFrame:
        """
        Aggregate warnings as a pandas Dataframe

        :param per_dataset: If true, warning counts for each dataset is returned
        :param as_counts: If true, get counts about warnings logged during extraction
            returns warning counts per song
        """
        entries = self.dset['music']

        def entry2df(d: Dict) -> pd.DataFrame:
            def prep_warn(d_warn: Dict) -> Dict:
                d_out = dict()
                d_out['src'] = d['title']
                d_out['type'] = d_warn.pop('warn_name', None)
                d_out['args'] = json.dumps(d_warn)
                d_out[self.key_dnm] = d[self.key_dnm]
                return d_out
            return pd.DataFrame([prep_warn(d) for d in d['warnings']])
        df = pd.concat([entry2df(e) for e in entries])
        if as_counts:
            def _get_counts(df_: pd.DataFrame, dnm: str = None) -> pd.DataFrame:
                counts = df_.type.value_counts()
                df_ = counts.to_frame(name='total_count').reset_index()  # Have `index` as a column
                df_.rename(columns={'index': 'type'}, inplace=True)
                df_['average_count'] = df_.apply(lambda x: x.total_count / self.n_song, axis=1)
                if dnm:
                    df_[self.key_dnm] = dnm
                return df_
            if per_dataset:
                return pd.concat([_get_counts(df[df[self.key_dnm] == dnm], dnm) for dnm in self.dnms])
            else:
                return _get_counts(df)
        else:
            return df

    def warning_type_dist(
            self, average=True, title: str = None, show_title: bool = True, show: bool = True, save: bool = False,
            **kwargs
    ):
        self.logger.info('Getting stats... ')
        df = self.warn_info(per_dataset=True)
        df_col2cat_col(df, 'type', WarnLog.types)
        typ = 'per piece' if average else 'in total'

        def callback(ax):
            severities = [WarnLog.type2severity[t] for t in WarnLog.types]
            cs = vals2colors(severities, color_palette='mako_r', gap=0.25)
            t2c = {t: cs[i] for i, t in enumerate(WarnLog.types)}

            plt.gcf().canvas.draw()
            ytick_lbs = ax.get_yticklabels()

            for t in ytick_lbs:
                txt = t.get_text()
                t.set_color(t2c[txt])
            ax.set_yticks(ax.get_yticks())  # disables warning
            ax.set_yticklabels([t.get_text() for t in ytick_lbs])  # Hack
        title_ = 'Distribution of Warnings during Music Extraction'
        if title is None and show_title:
            title = title_
        self.logger.info('Plotting... ')
        ax_ = barplot(
            data=df, x='type', y='average_count', title=title,
            xlabel='Warnings (color coded by severity)', ylabel=f'count {typ}',
            width=None, callback=callback, yscale='log', orient='h',
            hue=self.key_dnm if self.hue_by_dataset else None, show=not save, **kwargs
        )
        if save:
            save_fig(title_)
        if show:
            plt.show()
        return PlotOutputPair(df=df, ax=ax_)
        # TODO: set left to log scale and right to linear scale?
        # https://stackoverflow.com/questions/21746491/combining-a-log-and-linear-scale-in-matplotlib/21870368
        # solution here is real complicated


if __name__ == '__main__':
    from musicnlp.preprocess import dataset

    # md = 'melody'
    md = 'full'
    pch_kd = 'degree'

    pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD', as_full_path=True)
    # dnms, fnms = ['POP909'], [pop]
    # dnms, fnms = ['POP909', 'MAESTRO'], [pop, mst]
    dnms, fnms = ['POP909', 'MAESTRO', 'LMD'], [pop, mst, lmd]
    if dnms == ['POP909']:
        # cnm = f'22-03-12_MusViz-Cache_{{md={md[0]}, dnm=pop}}'
        cnm = f'22-04-05_MusViz-Cache_{{md={md[0]}, dnm=pop}}'
    elif dnms == ['POP909', 'MAESTRO']:
        cnm = f'22-04-05_MusViz-Cache_{{md={md[0]}, dnm=pop&mst}}'
    else:
        assert dnms == ['POP909', 'MAESTRO', 'LMD']
        cnm = f'22-04-09_MusViz-Cache_{{md={md[0]}}}, dnm=all-0.1}}'
    # cnm = None
    subset_ = 0.1 if 'LMD' in dnms else None  # LMD has 170k songs, prohibitive to plot all
    mv = MusicVisualize(
        filename=fnms, dataset_name=dnms, hue_by_dataset=True, cache=cnm, subset=subset_, pitch_kind=pch_kd
    )
    # mic(mv.df)

    def check_warn():
        df = mv.warn_info(as_counts=True)
        mic(df)
    # check_warn()

    def check_rare_tempos():
        tempos = mv.df.tempo.unique()
        mic(tempos)
        mic(set(tempos) - set(COMMON_TEMPOS))
    # check_rare_tempos()

    def check_rare_time_sigs():
        tss = mv.df.time_sig.unique()
        mic(tss)
        rare_tss = sorted(set(tss) - set(COMMON_TIME_SIGS), key=lambda ts: (ts[1], ts[0]))  # by denom first then numer
        mic(rare_tss)
    # check_rare_time_sigs()

    def plots():
        pd.set_option('display.max_rows', None)
        # plt.figure(figsize=(9, 4))
        # up = 97.7  # ~2 std on single side
        # up = 99.4  # 2.5 std on single side
        # up = 98.7  # 2.5 std on both sides
        up = 95
        args = dict(stat='percent', upper_percentile=up)
        # _args = dict(save=True, show_title=False)
        # args.update(_args)
        # mv.token_length_dist(**args)
        # mv.token_length_dist(tokenize_scheme='wordpiece', **args)
        mv.token_length_dist(tokenize_scheme='pairmerge', **args)

        # mv.song_duration_dist(**args)
        # mv.bar_count_dist(**args)

        # mic(mv.time_sig_dist(yscale='linear', stat='percent').df)
        # mv.tempo_dist(stat='percent')
        # mv.key_dist(stat='percent')

        # mv.note_pitch_dist(stat='percent')
        # mv.note_duration_dist(note_type='all', stat='percent')
        # mic(mv.note_duration_dist(note_type='tuplet', stat='percent').df)

        # mv.tuplet_count_dist(**args)
        # mv.tuplet_n_note_dist(**args)
        # mv.tuplet_duration_ratio()

        # mv.token_coverage_dist(ratio=0.99)
        # mv.empty_channel_ratio()
        # mv.rare_token_ratio()
        # mv.warning_type_dist(**_args)
    plots()

    def save_plots_for_report():
        plt.figure(figsize=(9, 5))
        ax = plt.gca()
        args = dict(new_figure=True, ax=ax, title=None)

        def token_len():
            mv.token_length_dist(**args)
            return 'Distribution of token length for each song'
        # title = token_len()

        def song_duration():
            mv.song_duration_dist(**args)
            return 'Distribution of song duration'
        # plt.show()
        title = song_duration()
        save_fig(f'{title}, {now(for_path=True)}')
    # save_plots_for_report()

    def save_for_report_warn():
        mv.warning_type_dist(title='None', show=False, bar_kwargs=dict(figure=plt.figure(figsize=(9, 5))))
        title = 'Average #Warnings for each song'
        save_fig(f'{title}, {now(for_path=True)}')
    # save_for_report_warn()

    def plots_for_presentation():
        # Colors from One Dark theme
        od_fg = hex2rgb('#B1B8C5', normalize=True)
        od_bg = hex2rgb('#282C34', normalize=True)
        od_blue = hex2rgb('#619AEF', normalize=True)
        od_purple = hex2rgb('#C678DD', normalize=True)

        plt.style.use('dark_background')
        sns.set(style='ticks', context='talk')
        plt.rcParams.update({
            'axes.facecolor': od_bg, 'figure.facecolor': od_bg, 'savefig.facecolor': od_bg,
            'xtick.color': od_fg, 'ytick.color': od_fg, 'axes.labelcolor': od_fg,
            'grid.linewidth': 0.5, 'grid.alpha': 0.5,
            'axes.linewidth': 0.5,
        })

        plt.figure(figsize=(9, 9))
        ax = plt.gca()
        args = dict(new_figure=False, ax=ax, title=None)

        def token_len():
            args['color'] = od_blue
            mv.token_length_dist(**args)
            return mv.df.n_token, 'Distribution of token length for each song'

        def song_duration():
            args.update(dict(color=od_purple, upper_percentile=97))
            mv.song_duration_dist(**args)
            return mv.df.duration, 'Distribution of song duration'

        vals, title = song_duration()
        save_fig(f'{title}, {now(for_path=True)}')
    # plots_for_presentation()

    def plot_for_report():
        mv.time_sig_dist(title=None, show=False)  # TODO: with bar plot instead?
        title = 'Distribution of Time Signature'
        save_fig(title)
    # plot_for_report()

    def check_bar_plots():
        # mv.time_sig_dist(kind='bar')
        mv.note_duration_dist(kind='hist')
    # check_bar_plots()
