import glob
import torch
import pathlib
import torchaudio
import torchvision
import torchlibrosa
import pandas as pd
from pathlib import PurePath
from torchcodec.decoders import AudioDecoder


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
            [self.metadata[i] for i in self.metadata.keys()]
        ).to_dict()['Annoyance']

        # set up list of files
        file_list = []
        for key, value in self.metadata.items():
            file_list.append([f'{input_path}/{key}/audio/{index}' 
                              for index in value.index])
        self.file_list = [y for x in file_list for y in x]
        
        self._set_up_transform()

        self.transformed_data = {}
        for file in self.file_list:
            x = AudioDecoder(file).get_all_samples().data
            self.transformed_data[file] = self._transform(x)
    
    def _set_up_transform(self):
        self._transform = torchaudio.transforms.MelSpectrogram(
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels)
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        filepath = self.file_list[index]
        x = self.transformed_data[filepath]

        # cast target to tensor
        y = self.targets[pathlib.Path(filepath).name]
        return x.squeeze(0), torch.tensor(y)


class MomentaryAnnoyanceLibrosa(MomentaryAnnoyance):
    def _set_up_transform(self):
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
        
        self._transform = torchvision.transforms.Compose([spectrogram, logmel])
