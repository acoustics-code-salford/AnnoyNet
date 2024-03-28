import torch
from torch import nn


class AnnoyNet(nn.Module):
    def __init__(self):
        super().__init__()

        # reusable relu
        self.relu = nn.ReLU()

        # conv layer 1
        self.conv1 = nn.Conv2d(1, 96, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 4))  # pool time dimension more
        self.bn1 = nn.BatchNorm2d(96)

        # conv layer 2
        self.conv2 = nn.Conv2d(96, 32, (3, 3))
        self.pool2 = nn.MaxPool2d((4, 8))  # pool time dimension more
        self.bn2 = nn.BatchNorm2d(32)

        # dense layers
        self.fc1 = nn.Linear(3_872, 1_000)
        self.fc2 = nn.Linear(1_000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.bn1(self.pool1(self.relu(self.conv1(x))))
        x = self.bn2(self.pool2(self.relu(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)