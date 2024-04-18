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


def plot_circumplex(dataframe):

    _, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

    axes.axhline(0, color='black', linewidth=0.9)
    axes.axvline(0, color='black', linewidth=0.9)

    axes.set_xlim((-1.1, 1.1))
    axes.set_ylim((-1.1, 1.1))

    circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=0.9)
    axes.add_patch(circ)

    axes.text(-1.18, -0.16, 'Unpleasant', rotation=90)
    axes.text(1.1, -0.13, 'Pleasant', rotation=270)
    axes.text(-0.13, 1.13, 'Aroused')
    axes.text(-0.09, -1.16, 'Calm')

    dataframe.plot.scatter('Valence', 'Arousal', ax=axes, s=10)

    plt.axis('off')


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
            n = outdim(n, 1, conv_kernel, pool_kernel)
    return n


def outsize(input_dimensions, n_layers, n_channels):
    dim1, dim2 = input_dimensions
    return outdim(dim1, n_layers) * outdim(dim2, n_layers) * n_channels
