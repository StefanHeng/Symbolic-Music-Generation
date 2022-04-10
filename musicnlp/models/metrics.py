import music21
from itertools import filterfalse
from collections import Counter

from musicnlp.vocab import VocabType, MusicConverter


def get_init_key_est(gt_token_seq: str, num_bars: int = 4):
    tok_lst = gt_token_seq.split()
    # Heuristics to determine starting bar
    key_est = None
    bar_idx = [idx for idx, tok in enumerate(
        tok_lst) if mc.vocab.type(tok) == VocabType.special]
    assert len(bar_idx) >= num_bars + 1, 'So many bars for extracting keys'

    pitch_lst = list(filterfalse(lambda x: mc.vocab.type(
        x) != VocabType.pitch, tok_lst[:bar_idx[num_bars +1]]))
    key_cls = [music21.pitch.Pitch(
        midi=mc.vocab.compact(p)).pitchClass for p in pitch_lst]
    key_est = Counter(key_cls).most_common()[0][0]
    return key_est


def get_eval_metric(gen_token_text: str, gt_token_text: str):
    tok_lst = gen_token_text.split()
    target_key = get_init_key_est(gt_token_text)
    pitch_lst = list(filterfalse(lambda x: mc.vocab.type(
        x) != VocabType.pitch, tok_lst))
    num_toks = len(pitch_lst)
    # Heuristics
    # TODO: add more music theories later
    key_thres, key_count, num_off_key = 8, 0, 0
    prev_p_cls = None
    num_off_key = 0
    for p in pitch_lst:
        p_cls = music21.pitch.Pitch(midi=mc.vocab.compact(p)).pitchClass
        if p_cls != target_key:
            num_off_key += 1
            # TODO: add cross-bar detection
            if prev_p_cls is not None and p_cls == prev_p_cls:
                key_count += 1
                if key_count >= key_thres:
                    target_key = p_cls
                    key_count = 0
            prev_p_cls = p_cls
        else:
            prev_p_cls = None
    return num_off_key / num_toks


if __name__ == '__main__':
    from icecream import ic

    import musicnlp.util.music as music_util

    mc = MusicConverter()

    def check_key_metric():
        text = music_util.get_extracted_song_eg(k='平凡之路')  # this one has tuplets
        ic(text[:200])
        ic(get_eval_metric(text, text))
    check_key_metric()
