import librosa
import matplotlib.pyplot as plt


def plot_spectrogram(specgram):
    _, ax = plt.subplots(1, 1)
    ax.set_ylabel('Frequency Bin')
    ax.set_xlabel('Frame')
    ax.imshow(librosa.power_to_db(specgram),
              origin="lower",
              aspect="auto",
              interpolation="nearest")
