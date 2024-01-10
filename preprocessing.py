import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import glob
import os


def plot_spectrogram(specgram,
                     title=None,
                     ylabel="freq_bin",
                     ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram),
              origin="lower",
              aspect="auto",
              interpolation="nearest")


# TODO: make transform user-definable
def process_to_mel(audio_dir, n_spec_frames=2344, outdir='melspecs/'):
    stimuli = glob.glob(f'{audio_dir}/*')

    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=48_000,
        n_mels=96,
        n_fft=1024,
        hop_length=512
    )

    for stimulus in stimuli:
        x, _ = torchaudio.load(stimulus)
        spec_x = melspec(x)
        if spec_x.shape[-1] > n_spec_frames:
            middle_cut_point = n_spec_frames//2
            spec_x = spec_x[...,
                            middle_cut_point:n_spec_frames+middle_cut_point]

        outpath = (outdir +
                   os.path.splitext(os.path.basename(stimulus))[0] +
                   '.pt')
        torch.save(spec_x, outpath)
        print('saved to', outpath)
