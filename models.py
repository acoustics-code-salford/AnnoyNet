import torch
from torch import nn


class AnnoyCNN(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()

        # reusable relu
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

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
        x = self.dropout(self.bn1(self.pool1(self.relu(self.conv1(x)))))
        x = self.dropout(self.bn2(self.pool2(self.relu(self.conv2(x)))))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return x


class AnnoyCRNN(nn.Module):
    def __init__(self, gru_layers=1, dropout_p=0.0):
        super().__init__()

        # reusable relu
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

        # conv layers
        self.conv1 = nn.Conv2d(1, 96, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 4))  # pool time dimension more
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 32, (3, 3))
        self.pool2 = nn.MaxPool2d((4, 8))  # pool time dimension more
        self.bn2 = nn.BatchNorm2d(32)

        # recurrent layer
        self.gru = nn.GRU(352, 128, gru_layers, dropout=dropout_p)

        # dense layers
        self.fc1 = nn.Linear(1_408, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.dropout(self.bn1(self.pool1(self.relu(self.conv1(x)))))
        x = self.dropout(self.bn2(self.pool2(self.relu(self.conv2(x)))))
        
        x = torch.flatten(x, 1, 2).movedim(-1, -2)  # stack over frequency axis
        x, _ = self.gru(x)
        
        x = torch.flatten(x, 1)  # flatten dimensions except batch
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x
