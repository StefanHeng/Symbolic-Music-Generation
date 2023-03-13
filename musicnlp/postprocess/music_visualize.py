import os
import re
import math
import json
import random
import pickle
from os.path import join as os_join
from copy import deepcopy
from typing import List, Tuple, Dict, Callable, Union
from fractions import Fraction
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


class MusicVisualize:
    """
    Visualize dataset info given json as extracted input representation

    See `preprocess.music_export.py`
    """
    key_dnm = 'dataset_name'
    color_rare = hex2rgb('#E06C75', normalize=True)
    pattern_frac = re.compile(r'^(?P<numer>\d+)/(?P<denom>\d+)$')

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

    def _set_meta(self):
        self.tokenizer = MusicTokenizer(precision=self.prec, pitch_kind=self.pitch_kind)
        self.wp_tokenizer = load_wordpiece_tokenizer(pitch_kind=self.pitch_kind, precision=self.prec)
        self.pm_tokenizer = load_pairmerge_tokenizer(pitch_kind=self.pitch_kind, precision=self.prec)

        assert self.wp_tokenizer.precision == self.prec
        self.vocab: MusicVocabulary = self.tokenizer.vocab
        self.stats = MusicStats(prec=self.prec, pitch_kind=self.pitch_kind)
        self.mc: MusicConverter = self.stats.converter
        sr_vocab = self.vocab if self.pitch_kind == 'step' else MusicVocabulary(pitch_kind='step', precision=self.prec)
        self.sr = transform.SanitizeRare(vocab=sr_vocab)
        # Another sanitize rare that preserves the original rare time sig & tempo
        self.sr_stat = transform.SanitizeRare(vocab=sr_vocab, rare_pitch_only=True)
        self.ak = transform.AugmentKey(vocab=self.vocab)

    def _extract_song_info(self, d: Dict, it=None) -> Dict:
        d = deepcopy(d)
        scr = d.pop('score')

        scr_ = self.sr(scr)
        keys = d['keys']
        # Needed for tokenization, just pick the most-confident key for simplicity
        scr_ = self.ak((scr_, max(keys, key=keys.get)))

        toks = self.tokenizer.tokenize(scr_)
        wp_toks = self.wp_tokenizer.tokenize(scr_, mode='char')  # just need len, efficient
        pm_toks = self.pm_tokenizer.tokenize(scr_)
        d['n_token'], d['n_wp_token'], d['n_pm_token'] = len(toks), len(wp_toks), len(pm_toks)

        elms = self.mc.str2music_elms(scr_, group=False, pitch_kind=self.pitch_kind).elms
        tups_ns = [len(e.meta[0]) for e in elms if e.type == ElmType.tuplets]  # number of pitch token
        # mic(tups_ns)
        c_tup_n_note = Counter(tups_ns)
        # if len(c_tup_n_note):
        #     mic(c_tup_n_note, tups_ns)
        #     raise NotImplementedError
        d['n_tuplet_note'] = dict(c_tup_n_note)

        if it:
            d_log = dict(n_token=pl.i(d['n_token']), n_wp_token=pl.i(d['n_wp_token']), n_pm_token=pl.i(len(pm_toks)))
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
        d['weighted_pitch_count'] = self.stats.weighted_pitch_counts(toks)
        return d

    def _get_song_info(self):
        entries: List[Dict] = self.dset['music']

        # concurrent = True
        concurrent = False
        if concurrent:
            tqdm_args = dict(desc='Extracting song info', unit='song', chunksize=64)
            ds = conc_map(self._extract_song_info, entries, with_tqdm=tqdm_args, mode='process')
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
            q = upper_percentile if isinstance(upper_percentile, float) else 99.7  # ~3std
            mi, ma = vs.min(), np.percentile(vs, q=q)
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

    def tuplet_count_dist(self, **kwargs):
        title = 'Distribution of #tuplets'
        args = dict(col_name='n_tup', title=title, xlabel='#tuplet', upper_percentile=True) | (kwargs or dict())
        return self.hist_wrapper(**args)

    def tuplet_n_note_dist(self, **kwargs):
        self.logger.info('Getting stats... ')
        k = 'n_tuplet_note'
        df = self._count_by_dataset(k)

        def callback(ax):
            import matplotlib.ticker as plticker

            loc = plticker.MultipleLocator(base=2)  # Use more frequent ticks cos the actual # matters
            ax.xaxis.set_major_locator(loc)
            # raise NotImplementedError
            ma, mi = df[k].max(), df[k].min()
            ax.set_xlim([mi, ma])
        title = 'Distribution of #notes in tuplets'
        return self.hist_wrapper(
            data=df, col_name='n_tuplet_note', weights='count', title=title, xlabel='#note in tuplets',
            discrete=True, kde=False, callback=callback, **kwargs
        )

    def song_duration_dist(self, **kwargs):
        def callback(ax):
            x_tick_vals = [v for v in plt.xticks()[0] if v >= 0]
            ax.set_xticks(x_tick_vals, labels=[sec2mmss(v) for v in x_tick_vals])
        title = 'Distribution of song duration'
        args = dict(col_name='duration', title=title, xlabel='duration (mm:ss)', callback=callback)
        if kwargs is not None:
            args.update(kwargs)
        return self.hist_wrapper(**args)

    def time_sig_dist(self, kind: str = 'hist', **kwargs):
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
            ax.set_xticks(ax.get_xticks())  # Hack to force rendering for `show`, TODO: better ways?
            ax.set_xticklabels([t.get_text() for t in xtick_lbs])
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
            return self.hist_wrapper(**args)
        else:
            df = self._count_by_dataset('time_sig')
            df['time_sig'] = df['time_sig'].map(lambda ts: f'{ts[0]}/{ts[1]}')  # so that the category ordering works
            args = dict(
                data=df, x='time_sig', y='count',
                hue=self.key_dnm, x_order=self.df.time_sig_str.cat.categories,
                xlabel='Time Signature', ylabel='count', title=title, yscale='log', width=False, callback=callback
            )
            args |= kwargs
            return barplot(**args)

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
        return self.hist_wrapper(**args)

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

    def note_duration_dist(self, kind='hist', **kwargs):
        """
        Tuplet notes contribute to a single duration, i.e. all quantized durations
        """
        self.logger.info('Getting stats... ')
        ca(dist_plot_type=kind)
        title, xlab = 'Distribution of Note Duration', 'Duration (quarter length)'
        if kind == 'hist':
            k = 'duration_count'
            df = self._count_by_dataset(k)
            df.rename(columns={k: 'duration'}, inplace=True)
            d_uniq = df.duration.unique()
            bound = min(d_uniq.max(), get_common_time_sig_duration_bound())

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
                ax.set_xticks(ax.get_xticks())  # disables warning
                ax.set_xticklabels([t.get_text() for t in xtick_lbs])  # Hack
            return self.hist_wrapper(
                data=df, col_name='duration', weights='count', discrete=True, kde=False,
                title=title, xlabel=xlab,
                callback=callback,
                **kwargs
            )
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

    @property
    def n_song(self) -> int:
        return len(self.dset['music'])

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
        return ax_
        # TODO: set left to log scale and right to linear scale?
        # https://stackoverflow.com/questions/21746491/combining-a-log-and-linear-scale-in-matplotlib/21870368
        # solution here is real complicated


if __name__ == '__main__':
    from musicnlp.preprocess import dataset

    # md = 'melody'
    md = 'full'
    pch_kd = 'degree'

    # dnms = ['POP909']
    # dnms = ['POP909', 'MAESTRO']
    dnms = ['POP909', 'MAESTRO', 'LMD']
    # dnms = ['LMD']
    pop, mst, lmd = dataset.get_dataset_dir_name('POP909', 'MAESTRO', 'LMD', as_full_path=True)
    # fnms = [pop]
    # fnms = [pop, mst]
    fnms = [pop, mst, lmd]
    if dnms == ['POP909']:
        cnm = f'22-03-12_MusViz-Cache_{{md={md[0]}, dnm=pop}}'
    elif dnms == ['POP909', 'MAESTRO']:
        cnm = f'22-03-12_MusViz-Cache_{{md={md[0]}, dnm=pop&mst}}'
    elif dnms == ['POP909', 'MAESTRO', 'LMD']:
        cnm = f'22-03-12_MusViz-Cache_{{md={md[0]}}}, dnm=all-0.1}}'
    else:
        cnm = None
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
        # plt.figure(figsize=(9, 4))
        args = dict(stat='percent', upper_percentile=97.7)  # ~2std
        # _args = dict(save=True, show_title=False)
        # args.update(_args)
        # mv.token_length_dist(**args)
        # mv.token_length_dist(tokenize_scheme='wordpiece', **args)
        # mv.token_length_dist(tokenize_scheme='pairmerge', **args)
        # mv.bar_count_dist(**args)
        # mv.tuplet_count_dist(**args)
        mv.tuplet_n_note_dist(**args)
        # mv.song_duration_dist(**args)
        # mv.time_sig_dist(yscale='linear', stat='percent')
        # mv.tempo_dist(stat='percent')
        # mv.key_dist(stat='percent')
        # mv.note_pitch_dist(stat='percent')
        # mv.note_duration_dist(stat='percent')
        # mv.warning_type_dist(**_args)
    plots()

    fig_sz = (9, 5)

    def save_plots_for_report():
        plt.figure(figsize=fig_sz)
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
        # plt.figure(figsize=(9, 5))
        mv.warning_type_dist(title='None', show=False, bar_kwargs=dict(figure=plt.figure(figsize=fig_sz)))
        title = 'Average #Warnings for each song'
        save_fig(f'{title}, {now(for_path=True)}')
    # save_for_report_warn()

    fig_sz = (9, 9)

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

        plt.figure(figsize=fig_sz)
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
