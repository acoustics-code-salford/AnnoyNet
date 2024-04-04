import librosa
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrogram(specgram):
    _, ax = plt.subplots(1, 1)
    ax.set_ylabel('Frequency Bin')
    ax.set_xlabel('Frame')
    ax.imshow(librosa.power_to_db(specgram),
              origin="lower",
              aspect="auto",
              interpolation="nearest")

def array_chunk_split(array, chunk_size, stride=None):
    if not stride:
        stride = chunk_size
    return np.lib.stride_tricks.sliding_window_view(
        array, chunk_size, axis=0)[::stride]
