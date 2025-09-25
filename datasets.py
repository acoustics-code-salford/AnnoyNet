import os
import glob
import torch
import torchaudio
import torchvision
import torchlibrosa
import pandas as pd
from pathlib import PurePath


class MomentaryAnnoyance(torch.utils.data.Dataset):
    def __init__(self,
                 input_path,
                 n_fft=512,
                 hop_length=None,
                 n_mels=100,
                 fs=16_000,
                 select_folds=None):
        
        if not hop_length:
            hop_length = n_fft // 2
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.fs = fs
        self.n_time_frames = int(1 / self.hop_length * 6 * self.fs + 1)
        self.input_path = input_path

        # load metadata into dict
        metadata_filepaths = glob.glob(f'{input_path}/*/*.csv')
        self.metadata = {}
        for path in metadata_filepaths:
            dataset_key = PurePath(path).parts[-2]
            subset_meta = pd.read_csv(path, index_col=0)

            # select data from specified folds
            if select_folds:
                query = ' or '.join([f'fold == {x}' for x in select_folds])
                self.metadata[dataset_key] = subset_meta.query(query)
            else:
                self.metadata[dataset_key] = subset_meta

        # filter selected datasets by key list
        self.targets = pd.concat(
            [self.metadata[i] for i in self.metadata.keys()])

        # set up list of files
        file_list = []
        for key, value in self.metadata.items():
            file_list.append([f'{input_path}/{key}/audio/{index}' 
                              for index in value.index])
        self.file_list = [y for x in file_list for y in x]
        
        self._set_up_transforms()
    
    def _set_up_transforms(self):
        self.resample = torchaudio.transforms.Resample(
            48_000, self.fs)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels)
        self.transform = torchvision.transforms.Compose([
            self.resample, self.melspec])
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        filepath = self.file_list[index]
        x, _ = torchaudio.load(filepath)

        # sum to mono and apply transforms
        x = self.transform(x.sum(0).unsqueeze(0)).squeeze()
        
        # zero-pad if less than 6 seconds of frames
        x = torch.concatenate(
            (x, torch.zeros((self.n_time_frames - len(x), self.n_mels)))
        )
        # unsqueeze data to add channel dimension
        x = x.unsqueeze(0)

        # cast target to tensor
        y = self.targets.loc[os.path.basename(filepath)].values[0]
        y = torch.tensor(y).float()
        return x, y


class MomentaryAnnoyanceLibrosa(MomentaryAnnoyance):
    def _set_up_transforms(self):
        resample = torchaudio.transforms.Resample(
            48_000, self.fs)

        spectrogram = torchlibrosa.stft.Spectrogram(
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.n_fft,
            window='hann',
            center=True,
            pad_mode='reflect',
            freeze_parameters=True)

        logmel = torchlibrosa.stft.LogmelFilterBank(
            sr=self.fs,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=50,
            fmax=self.fs // 2,
            ref=1.0,
            amin=1e-10,
            freeze_parameters=True)
        
        self.transform = torchvision.transforms.Compose([
            resample, spectrogram, logmel])
