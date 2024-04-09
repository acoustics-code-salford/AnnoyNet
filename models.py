import torch
from torch import nn
from utils import outdim


class AnnoyCNN(nn.Module):
    def __init__(self, input_shape, dropout=0.2, 
                 conv_kernel=(3, 3), maxpool_kernel=(2, 8)):
        super().__init__()

        n_freqdim_out = outdim(input_shape[0], 2, 
                               conv_kernel[0], 
                               maxpool_kernel[0])
        n_timedim_out = outdim(input_shape[1], 2, 
                               conv_kernel[1],
                               maxpool_kernel[1])
        
        # 32 output channels on last conv layer
        linear_input_len = n_freqdim_out * n_timedim_out * 32

        # reusable modules
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # conv layer 1
        self.conv1 = nn.Conv2d(1, 96, conv_kernel)
        self.pool1 = nn.MaxPool2d(maxpool_kernel)
        self.bn1 = nn.BatchNorm2d(96)

        # conv layer 2
        self.conv2 = nn.Conv2d(96, 32, conv_kernel)
        self.pool2 = nn.MaxPool2d(maxpool_kernel)
        self.bn2 = nn.BatchNorm2d(32)

        # dense layers
        self.fc1 = nn.Linear(linear_input_len, 1_000)
        self.fc2 = nn.Linear(1_000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.dropout(self.bn1(self.pool1(self.relu(self.conv1(x)))))
        x = self.dropout(self.bn2(self.pool2(self.relu(self.conv2(x)))))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = torch.clip(self.relu(self.fc3(x)), max=10)
        return x


class AnnoyCRNN(nn.Module):
    def __init__(self, input_shape, gru_layers=2, dropout=0.2,
                 conv_kernel=(3, 3), maxpool_kernel=(2, 8)):
        super().__init__()

        n_freqdim_out = outdim(input_shape[0], 2, 
                               conv_kernel[0], 
                               maxpool_kernel[0])
        n_timedim_out = outdim(input_shape[1], 2, 
                               conv_kernel[1],
                               maxpool_kernel[1])
        
        # 32 output channels on last conv layer
        linear_input_len = n_freqdim_out * n_timedim_out * 32
        gru_input_len = n_freqdim_out * 32

        # reusable relu
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # conv layers
        self.conv1 = nn.Conv2d(1, 96, conv_kernel)
        self.pool1 = nn.MaxPool2d(maxpool_kernel)
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 32, conv_kernel)
        self.pool2 = nn.MaxPool2d(maxpool_kernel)
        self.bn2 = nn.BatchNorm2d(32)

        # recurrent layer
        self.gru = nn.GRU(gru_input_len, 128, gru_layers, dropout=dropout)

        # dense layers
        self.fc1 = nn.Linear(linear_input_len, 1_000)
        self.fc2 = nn.Linear(1_000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.dropout(self.bn1(self.pool1(self.relu(self.conv1(x)))))
        x = self.dropout(self.bn2(self.pool2(self.relu(self.conv2(x)))))
        
        x = torch.flatten(x, 1, 2).movedim(-1, -2)  # stack over frequency axis
        x, _ = self.gru(x)
        
        x = torch.flatten(x, 1)  # flatten dimensions except batch
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = torch.clip(self.relu(self.fc3(x)), max=10)
        return x


class AnnoyCRNN_4(nn.Module):
    def __init__(self, gru_layers=2, dropout=0.0):
        super().__init__()

        # reusable relu
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # conv layers
        self.conv1 = nn.Conv2d(1, 96, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 4))  # pool time dimension more
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 32, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 4))  # pool time dimension more
        self.bn2 = nn.BatchNorm2d(32)

        # recurrent layer
        self.gru = nn.GRU(128, 128, gru_layers, dropout=dropout)  # 352

        # dense layers
        self.fc1 = nn.Linear(11_904, 1_000)  # 1_408
        self.fc2 = nn.Linear(1_000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.dropout(self.bn1(self.pool1(self.relu(self.conv1(x)))))
        x = self.dropout(self.bn2(self.pool2(self.relu(self.conv2(x)))))
        
        x = torch.flatten(x, 1, 2).movedim(-1, -2)  # stack over frequency axis
        x, _ = self.gru(x)
        
        x = torch.flatten(x, 1)  # flatten dimensions except batch
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return x


class AnnoyCRNN_2(nn.Module):
    def __init__(self, gru_layers=2, dropout=0.0):
        super().__init__()

        # reusable relu
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # conv layers
        self.conv1 = nn.Conv2d(1, 96, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 32, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.bn2 = nn.BatchNorm2d(32)

        # recurrent layer
        self.gru = nn.GRU(128, 128, gru_layers, dropout=dropout)  # 352

        # dense layers
        self.fc1 = nn.Linear(47_744, 1_000)  # 1_408
        self.fc2 = nn.Linear(1_000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.dropout(self.bn1(self.pool1(self.relu(self.conv1(x)))))
        x = self.dropout(self.bn2(self.pool2(self.relu(self.conv2(x)))))
        
        x = torch.flatten(x, 1, 2).movedim(-1, -2)  # stack over frequency axis
        x, _ = self.gru(x)
        
        x = torch.flatten(x, 1)  # flatten dimensions except batch
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return x
