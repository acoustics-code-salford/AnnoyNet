import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from collections import OrderedDict


# def init_layer(layer):
#     """Initialize a Linear or Convolutional layer. """
#     nn.init.xavier_uniform_(layer.weight)
 
#     if hasattr(layer, 'bias'):
#         if layer.bias is not None:
#             layer.bias.data.fill_(0.)
            
    
# def init_bn(bn):
#     """Initialize a Batchnorm layer. """
#     bn.bias.data.fill_(0.)
#     bn.weight.data.fill_(1.)
    

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

    #     self.init_weight()
        
    # def init_weight(self):
    #     init_layer(self.conv1)
    #     init_layer(self.conv2)
    #     init_bn(self.bn1)
    #     init_bn(self.bn2)

        
    def forward(self, input):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = self.pool(x)

        return x
    

class Cnn14(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn14, self).__init__()

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, 
            time_stripes_num=2, 
            freq_drop_width=8, 
            freq_stripes_num=2
        )

        self.bn0 = nn.BatchNorm2d(64)

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

        clipwise_output = self.fc_block(x)

        return clipwise_output