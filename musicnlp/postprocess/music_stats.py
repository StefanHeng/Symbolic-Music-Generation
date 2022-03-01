from collections import Counter

from musicnlp.util import *
from musicnlp.util.music import Dur
from musicnlp.preprocess import MusicVocabulary


class MusicStats:
    def __init__(self, prec: int = 5):
        self.prec = prec
        self.vocab = MusicVocabulary(prec=prec, color=False)

    def vocab_type_counts(self, toks: Iterable[str]) -> Dict[str, Counter]:
        """
        :param toks: Iterable of token strings
        :return: Counter on compact representation by the compact token types
        """
        toks = sorted(toks)
        type2toks = {k: list(v) for k, v in itertools.groupby(toks, key=lambda tok: self.vocab.type(tok))}
        type2toks = {
            k: list(self.vocab.compact(t) for t in v) for k, v in type2toks.items() if k in self.vocab.compacts
        }
        return {k.name: Counter(v) for k, v in type2toks.items()}

    def weighted_pitch_counts(self, toks: Iterable[str]) -> Dict[str, Dur]:
        """
        :return: counts for pitch, weighed by the duration in quarter length
        """


if __name__ == '__main__':
    from icecream import ic

    fnm = 'musicnlp music extraction, dnm=POP909, n=909, mode=melody, 2022-02-25 20-59-06'
    ms = MusicStats()

    def check_vocab_type():
        with open(os.path.join(config('path-export'), f'{fnm}.json')) as f:
            text = json.load(f)['music'][0]['text']
        toks = text.split()

        ic(ms.vocab_type_counts(toks))
    check_vocab_type()
