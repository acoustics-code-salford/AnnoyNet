import torch
from torch import nn


class VGGBinaural(nn.Module):
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
