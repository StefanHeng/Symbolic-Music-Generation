from stefutil import *
import musicnlp.util.music as music_util
from musicnlp.vocab import MusicTokenizer
from musicnlp.preprocess import MusicExtractor, transform


if __name__ == '__main__':
    def viz_train_aug():
        me = MusicExtractor(
            mode='full', with_pitch_step=True, greedy_tuplet_pitch_threshold=16, warn_logger=True, verbose=False
        )

        fnm = 'Merry Go Round of Life'
        fnm = music_util.get_my_example_songs(fnm, fmt='MXL')
        d_out = me(fnm, exp='str_join', return_meta=True, return_key=True)
        mic(d_out)

        text, key = d_out.score, max(d_out.keys, key=d_out.keys.get)
        mic(text, key)

        ak = transform.AugmentKey()
        text = ak((text, key))
        mic(text)

        tokenizer = MusicTokenizer(pitch_kind='degree')
        text = tokenizer.colorize(text)
        print(text)
    viz_train_aug()
