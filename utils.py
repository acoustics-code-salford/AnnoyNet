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


def outdim(n, n_layers=1, conv_kernel=3, pool_kernel=2):
    if n_layers == 1:
        # incorporates conv kernel and pooling
        n = ((((n - conv_kernel) + 1) - pool_kernel) // pool_kernel) + 1
    else:
        for i in range(n_layers):
            n = outdim(n)
    return n


def outsize(input_dimensions, n_layers, n_channels):
    dim1, dim2 = input_dimensions
    return outdim(dim1, n_layers) * outdim(dim2, n_layers) * n_channels