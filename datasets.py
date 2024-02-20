from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import glob
import pandas as pd
import torchaudio, torchvision


class DronePeakMFCCAnnoyance(torch.utils.data.Dataset):

    def __init__(self,
                 input_path='raw_data/',
                 targets_file='mean_annoyances.csv'):

        self.file_list = glob.glob(f'{input_path}*')
        self.targets = pd.read_csv(targets_file, index_col=0)
        self.resample = torchaudio.transforms.Resample(48_000, 16_000)
        self.mfcc = torchaudio.transforms.MFCC(
            n_mfcc=20,
            melkwargs={
                "n_fft": 400,
                "hop_length": 4096, # 1/5th second
                "n_mels": 20,
                "center": False
            }
        )
        self.transform = torchvision.transforms.Compose([
            self.resample, self.mfcc])

        # silly trick to root out invalid (too short) clips
        self._use_valid_idx = False
        self.valid_idx = [i for i, (x, _) in enumerate(self) if len(x) == 24]
        self._use_valid_idx = True
        # definitely not advisable for larger datasets

    def __len__(self):
        # return len(self.file_list)
        return len(self.valid_idx)

    def __getitem__(self, index):
        if self._use_valid_idx:
            valid_idx = self.valid_idx[index]
            filepath = self.file_list[valid_idx]
        else:
            filepath = self.file_list[index]
        x, _ = torchaudio.load(filepath)
        x = x.sum(0)
        x = self.transform(x)

        # select peak frame
        max_0 = torch.where(x[0] == torch.max(x[0]))[0][0]

        # mfccs from peak frame +- 3 seconds(ish)
        x = x[:, max_0-12:max_0+12].T

        index_str = os.path.basename(filepath)
        y = (
            (torch.tensor(
                self.targets.loc[index_str].values[0],
                dtype=torch.float32) / 7) - 0.5  # scale += 0.5
        )  # scale between 0 and 1

        return x, y.repeat(len(x), 1)  # repeat y across frames


class DronePeakMFCCAffect(torch.utils.data.Dataset):

    def __init__(self,
                 input_path='raw_data/',
                 targets_file='mean_ratings.csv'):

        self.targets = pd.read_csv(targets_file, index_col=0)
        self.file_list = [f'{input_path}{f}.wav' for f in self.targets.index]
        self.resample = torchaudio.transforms.Resample(48_000, 16_000)
        self.mfcc = torchaudio.transforms.MFCC(
            n_mfcc=20,
            melkwargs={
                "n_fft": 400,
                "hop_length": 4096, # 1/5th second
                "n_mels": 20,
                "center": False
            }
        )
        self.transform = torchvision.transforms.Compose([
            self.resample, self.mfcc])

        # # silly trick to root out invalid (too short) clips
        self._use_valid_idx = False
        self.valid_idx = [i for i, (x, _) in enumerate(self) if len(x) == 24]
        self._use_valid_idx = True
        # # definitely not advisable for larger datasets

    def __len__(self):
        # return len(self.file_list)
        return len(self.valid_idx)

    def __getitem__(self, index):
        if self._use_valid_idx:
            valid_idx = self.valid_idx[index]
            filepath = self.file_list[valid_idx]
        else:
            filepath = self.file_list[index]
        x, _ = torchaudio.load(filepath)
        x = x.sum(0)
        x = self.transform(x)

        # select peak frame
        max_0 = torch.where(x[0] == torch.max(x[0]))[0][0]

        # mfccs from peak frame +- 3 seconds(ish)
        x = x[:, max_0-12:max_0+12].T

        index_str = (
            os.path.basename(
            filepath).split('.')[0]
            .split('_C')
        )[0]  # THIS WILL NOW BE BROKEN
        y = (((torch.tensor(self.targets.loc[index_str].values,
                            dtype=torch.float32)
                - torch.tensor([1, 1, 0])) # scale targets petween +- 0.5
                / torch.tensor([4, 4, 10]))
                - torch.tensor([.5, .5, .5]))
        # this might not strictly be the best thing to do with annoyance as
        # there's no neutral central spot for this metric

        return x, y.repeat(len(x), 1) # repeat y across frames


class UAVNoiseAffectMels(Dataset):

    def __init__(self,
                 data_path='melspecs/',
                 targets_file='mean_ratings.csv',
                 transforms=None):

        self.transforms = transforms
        self.file_list = glob.glob(f'{data_path}*')
        self.targets = pd.read_csv(targets_file, index_col=0)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        # load file
        filepath = self.file_list[item]
        x = torch.load(filepath)

        index_str = (
            os.path.basename(
            filepath).split('.')[0]
            .split('_C')
        )[0]
        y = self.targets.loc[index_str].values

        # probably no transforms (data already processed) but good to include
        if self.transforms is not None:
            x = self.transforms(x)

        return x, y
