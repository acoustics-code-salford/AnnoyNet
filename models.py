import torch
from torch import nn


class VariCRNN(nn.Module):
    def __init__(self, 
                 n_conv_layers=1, 
                 n_feature_maps=32, 
                 n_gru_layers=1,
                 gru_hidden_size=128,
                 dropout=0.0):
        super().__init__()
        self.n_gru_layers = n_gru_layers

        # reusable modules
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d((2, 2))
        
        self.conv1 = nn.Conv2d(1, n_feature_maps, (3, 3))
        self.bn1 = nn.BatchNorm2d(n_feature_maps)

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(n_feature_maps, n_feature_maps, (3, 3))
            for i in range(n_conv_layers-1)])
        self.batchnorms = nn.ModuleList([
            nn.BatchNorm2d(n_feature_maps) for i in range(n_conv_layers-1)
        ])
        
        n_out_features = self.outdim(100, n_conv_layers) * \
            self.outdim(376, n_conv_layers) * n_feature_maps
        
        if n_gru_layers:
            len_gru_input = self.outdim(100, n_conv_layers) * n_feature_maps
            self.gru = nn.GRU(len_gru_input, gru_hidden_size, n_gru_layers, 
                              dropout=dropout)

        # dense layers
        self.fc1 = nn.Linear(n_out_features, 1_000)
        self.fc2 = nn.Linear(1_000, 100)
        self.fc3 = nn.Linear(100, 1)
        
    def outdim(self, n, n_layers=1):
        if n_layers == 1:
            # incorporates conv kernel and pooling
            n = ((((n - 3) + 1) - 2) // 2) + 1
        else:
            for i in range(n_layers):
                n = self.outdim(n)
        return n
    
    def forward(self, x):
        x = self.dropout(self.bn1(self.pool(self.relu(self.conv1(x)))))
        for conv, bn in zip(self.conv_layers, self.batchnorms):
            x = self.dropout(bn(self.pool(self.relu(conv(x)))))

        if self.n_gru_layers:
            # stack over frequency axis
            x = torch.flatten(x, 1, 2).movedim(-1, -2)
            x, _ = self.gru(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return x


class AnnoyCNN(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()

        # reusable modules
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

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
    def __init__(self, gru_layers=1, dropout=0.0):
        super().__init__()

        # reusable relu
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # conv layers
        self.conv1 = nn.Conv2d(1, 96, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 4))  # pool time dimension more
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 32, (3, 3))
        self.pool2 = nn.MaxPool2d((4, 8))  # pool time dimension more
        self.bn2 = nn.BatchNorm2d(32)

        # recurrent layer
        self.gru = nn.GRU(352, 128, gru_layers, dropout=dropout)

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
