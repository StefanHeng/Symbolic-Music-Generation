"""
Modified from https://stackoverflow.com/a/61975760/10732321

Works only in command line
Rerun this script for a new midi
"""

import pygame
from random import randint

from musicnlp.util.pkg_paths import *


if __name__ == '__main__':
    paths = get_midi_paths('LMD_matched')  # Specify pool of Midi file paths
    n = len(paths)
    idx = randint(0, n-1)

    fnm = paths[idx]
    fnm_ = fnm[len(BASE_PATH):]

    def play_midi():
        freq = 44100
        bitsize = -16
        channels = 2
        buffer = 1024
        pygame.mixer.init(freq, bitsize, channels, buffer)
        pygame.mixer.music.set_volume(0.8)

        def _play():
            """
            Stream music_file in a blocking manner
            """
            clock = pygame.time.Clock()
            pygame.mixer.music.load(fnm)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                clock.tick(30)
        try:
            print(f'Playing midi file {idx+1} of {n}: {fnm_}')
            _play()
        except KeyboardInterrupt:
            pygame.mixer.music.fadeout(1000)
            pygame.mixer.music.stop()
            raise SystemExit

    play_midi()
