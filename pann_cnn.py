import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from collections import OrderedDict
   

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_shape, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.AdaptiveAvgPool2d(out_shape)
        
    def forward(self, input):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = self.pool(x)

        return x


def basic_conv_block(in_chans, out_chans, 
               kernel_size=(5, 5), 
               out_shape=(2, 2)):
    
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_chans, out_chans,
                            kernel_size=kernel_size, 
                            stride=(1, 1),
                            padding=(2, 2), 
                            bias=False)),
        ('bn1', nn.BatchNorm2d(out_chans)),
        ('relu', nn.ReLU()),
        ('avgpool', nn.AdaptiveAvgPool2d(out_shape)),
        ('dropout', nn.Dropout(p=0.2))
    ]))


class LSTMBlock(nn.Module):
    def __init__(self, input_size):
        
        super(LSTMBlock, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, 128,
                         batch_first=True,
                         bidirectional=True)
                              
        self.lstm2 = nn.LSTM(256, 128,
                         batch_first=True,
                         bidirectional=True)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        return x
    

class Cnn6(nn.Module):
    def __init__(self, classes_num, n_mels=64):
        
        super(Cnn6, self).__init__()

        self.spec_augmenter = SpecAugmentation(time_drop_width=64, 
                                               time_stripes_num=2, 
                                               freq_drop_width=8, 
                                               freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(n_mels)

        self.conv_block1 = basic_conv_block(1, 64, out_shape=(350, 32))
        self.conv_block2 = basic_conv_block(64, 128, out_shape=(175, 16))
        self.conv_block3 = basic_conv_block(128, 256, out_shape=(87, 8))
        self.conv_block4 = basic_conv_block(256, 512, out_shape=(43, 4))

        self.fc_block = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(p=0.5)),
            ('fc1', nn.Linear(512, 512)),
            ('relu', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(512, classes_num)),
            ('sigmoid', nn.Sigmoid())]
        ))
 
    def forward(self, x):
        """
        Input: (batch_size, data_length)"""
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        output = self.fc_block(x)

        return output


class Cnn14(nn.Module):
    def __init__(self, classes_num, n_mels=64):
        
        super(Cnn14, self).__init__()

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, 
            time_stripes_num=2, 
            freq_drop_width=8, 
            freq_stripes_num=2
        )

        self.bn0 = nn.BatchNorm2d(n_mels)

        self.conv_block1 = ConvBlock(out_shape=(350, 52), 
                                     in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(out_shape=(175, 16), 
                                     in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(out_shape=(87, 8), 
                                     in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(out_shape=(43, 4), 
                                     in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(out_shape=(21, 2), 
                                     in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(out_shape=(21, 2), 
                                     in_channels=1024, out_channels=2048)

        self.fc_block = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(p=0.5)),
            ('fc1', nn.Linear(2048, 2048)),
            ('relu', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(2048, classes_num)),
            ('sigmoid', nn.Sigmoid())]
        ))
 
    def forward(self, x):
        """
        Input: (batch_size, data_length)"""

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        output = self.fc_block(x)

        return output
    

def replace_fc_layer():
    model = Cnn14(527)
    checkpoint = torch.load('pann_cnn14_pretrained.pth', map_location=device)
    model.load_state_dict(checkpoint)

    # replace classification block with a regression block
    for param in model.parameters():
        param.requires_grad = False

    # classification block from internoise CNN model
    new_fc_block = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 1000)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(1000, 100)), 
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.2)),
        ('fc3', nn.Linear(100, 1)),
        ('output', nn.ReLU())
    ]))

    model.fc_block = new_fc_block
    for layer in model.fc_block:
        if isinstance(layer, (nn.Linear)):
            torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
    return model