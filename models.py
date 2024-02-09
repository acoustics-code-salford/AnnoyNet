import torch
from torch import nn
import pytorch_lightning as pl


class VGGBinaural(pl.LightningModule):
    '''
    Modified version of VGG-like network described in FSD50K paper
    This version collapses the time dimension so that the global pooling 
    operation is not so asymmetric
    '''

    def __init__(self):
        super().__init__()

        self.conv_group1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 4))
        )

        self.conv_group2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 4))
        )

        self.conv_group3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 4))
        )

        self.glob_maxpool = nn.MaxPool2d(kernel_size=(12, 36))
        self.glob_avgpool = nn.AvgPool2d(kernel_size=(12, 36))

        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, 3)
        )

    def forward(self, x):

        x = self.conv_group1(x)
        x = self.conv_group2(x)
        x = self.conv_group3(x)

        x_max = self.glob_maxpool(x).squeeze()
        x_avg = self.glob_avgpool(x).squeeze()

        # in case 'batch' of one is missing first dimension
        if x_max.ndim < 2:
            x_max = x_max.unsqueeze(0)
            x_avg = x_avg.unsqueeze(0)

        x = torch.cat((x_max, x_avg), 1)

        x = self.fc_layers(x)

        return torch.sigmoid(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def mse_loss(self, y_pred, y_true):
        loss = nn.MSELoss()
        return loss(y_pred, y_true)
    
    def training_step(self, train_batch):
        x, y_true = train_batch
        y_pred = self.forward(x)
        loss = self.mse_loss(y_pred, y_true)
        self.log('train_loss', loss)

    def validation_step(self, val_batch):
        x, y_true = val_batch
        y_pred = self.forward(x)
        loss = self.mse_loss(y_pred, y_true)
        self.log('val_loss', loss)


# trainer = pl.Trainer()
# trainer.fit(VGGBinaural(), datamodule=UAVNoiseDataModule())
        

class SimpleDenseLSTM(nn.Module):
    def __init__(self, input_size, out_size=3, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.dense = nn.Linear(hidden_size, out_size)

    def forward(self, input):
        outputs, _ = self.lstm(input)
        outputs = self.dense(outputs)
        return outputs