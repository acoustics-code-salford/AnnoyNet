import os
import glob
import torch
import torchaudio
import torchvision
import pandas as pd
from pathlib import PurePath


class MomentaryAnnoyance(torch.utils.data.Dataset):
    def __init__(self,
                 input_path,
                 n_mels=100,
                 key_select=None):

        self.input_path = input_path

        # load metadata into dict
        metadata_filepaths = glob.glob(f'{input_path}/*/*.csv')
        metadata = {}
        for path in metadata_filepaths:
            dataset_key = PurePath(path).parts[-2]
            metadata[dataset_key] = pd.read_csv(path, index_col=0)

        # select all dataset keys if not specified
        key_select = metadata.keys() if not key_select else key_select
        
        # filter selected datasets by key list
        self.targets = pd.concat([metadata[i] for i in key_select])

        # set up list of files
        self.file_list = []
        for key in key_select:
            for file in glob.glob(f'{input_path}/{key}/audio/*'):
                self.file_list.append(file)

        # set up transforms
        self.resample = torchaudio.transforms.Resample(48_000, 16_000)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            n_fft=512, n_mels=n_mels)
        self.transform = torchvision.transforms.Compose([
            self.resample, self.melspec])
        
        self.n_mels = n_mels
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        filepath = self.file_list[index]
        x, _ = torchaudio.load(filepath)

        # sum to mono and apply transforms
        x = x.sum(0)
        x = self.transform(x).T
        # zero-pad if less than 376 frames (6 seconds)
        x = torch.concatenate((x, torch.zeros((376 - len(x), self.n_mels)))).T
        # unsqueeze data to add channel dimension
        x = x.unsqueeze(0)

        # cast target to tensor
        y = self.targets.loc[os.path.basename(filepath)].values[0]
        y = torch.tensor(y).float()

        return x, y