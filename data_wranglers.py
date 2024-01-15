from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torch
import os
import glob
import pandas as pd


class UAVNoiseAffect(Dataset):

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


class UAVNoiseDataModule(pl.LightningDataModule):

    def setup(self, stage):
        self.dataset = UAVNoiseAffect()
        self.train_data, self.val_data = random_split(self.dataset, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(self.train_data, 1, num_workers=19)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, 1, num_workers=19)