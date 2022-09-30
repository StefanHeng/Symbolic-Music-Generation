
import matplotlib.pyplot as plt
import seaborn as sns
import pretty_midi
import librosa
from librosa import display
from mido import MidiFile

sns.set_style('darkgrid')


if __name__ == '__main__':
    from stefutil import mic

    path = '/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/MIDI-eg/' \
           'Ed Sheeran - Shape of You (Carlo Prato).mid'
    mid = MidiFile(path, clip=True)
    # mic(mid)


    def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
        librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                                 hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                                 fmin=pretty_midi.note_number_to_hz(start_pitch))

    pm = pretty_midi.PrettyMIDI(path)
    plt.figure(figsize=(8, 4))
    plot_piano_roll(pm, 56, 70)

