import torch
from torch import nn
from utils import outdim
from collections import OrderedDict
from pann_cnn import LSTMBlock


class AnnoyCnn(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.conv_block1 = self.conv_block(1, 96, **kwargs)
        self.conv_block2 = self.conv_block(96, 32, **kwargs)

        # dense layers
        self.fc_block = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(32*23*22, 1000)), #14
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(1000, 100)), 
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(100, 1)),
            ('output', nn.ReLU())
        ]))
    
    def conv_block(self, in_chans, out_chans,
               conv_kernel=(3, 3),
               maxpool_kernel=(8, 2),
               dropout_p=0.2):
    
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_chans, out_chans, conv_kernel)),
            ('relu1', nn.ReLU()),
            ('pool', nn.AvgPool2d(maxpool_kernel)),
            ('batchnorm', nn.BatchNorm2d(out_chans)),
            ('dropout', nn.Dropout(dropout_p))
        ]))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = torch.flatten(x, 1)
        x = self.fc_block(x)
        return torch.clip(x, max=10)


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
        linear_input_len = n_timedim_out * 128  # gru_output_len
        gru_input_len = n_freqdim_out * 32  # n_filters
        
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

        self.fc_block = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2048, 1000)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(1000, 100)), 
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(100, 1)),
            ('output', nn.ReLU())
        ]))

    def forward(self, x):
        x = self.dropout(self.bn1(self.pool1(self.relu(self.conv1(x)))))
        x = self.dropout(self.bn2(self.pool2(self.relu(self.conv2(x)))))
        
        x = torch.flatten(x, 1, 2).movedim(-1, -2)  # stack over frequency axis
        x, _ = self.gru(x)
        
        x = torch.flatten(x, 1)  # flatten dimensions except batch
        x = self.fc_block(x)
        x = torch.clip(x, max=10)
        return x


class AnnoyCnn3(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block1 = self.conv_block(1, 128)#, out_shape=(187, 25))
        self.conv_block2 = self.conv_block(128, 64)#, out_shape=(90, 10))
        self.conv_block3 = self.conv_block(64, 32)#, out_shape=(45, 2))

        # dense layers
        self.fc_block = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(32*6*10, 1000)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(1000, 100)), 
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(100, 1)),
            ('output', nn.ReLU())
        ]))
    
    def conv_block(self, in_chans, out_chans,# out_shape,
               conv_kernel=(3, 3),
               dropout_p=0.2):
    
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_chans, out_chans, conv_kernel)),
            ('relu1', nn.ReLU()),
            ('pool', nn.AvgPool2d((6, 2))),
            ('batchnorm', nn.BatchNorm2d(out_chans)),
            ('dropout', nn.Dropout(dropout_p))
        ]))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x, 1)
        x = self.fc_block(x)
        return torch.clip(x, max=10)


class AnnoyCrnn3(AnnoyCnn3):
    def __init__(self):
        super().__init__()
        self.lstm_block = LSTMBlock(1920)
        # dense layers
        self.fc_block = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(256, 1000)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(1000, 100)), 
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(100, 1)),
            ('output', nn.ReLU())
        ]))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x, 1)
        x = self.lstm_block(x)
        x = self.fc_block(x)
        return torch.clip(x, max=10)