import torch
import glob, os
import numpy as np
import pandas as pd
import torchaudio, torchvision

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from pytorchtools import EarlyStopping


def training_loop(n_epochs,
                  optimiser, 
                  model, 
                  loss_fn, 
                  dataloader, 
                  patience=20):
    
    train_losses = []
    avg_train_losses = []
    early_stopping = EarlyStopping(patience=patience, 
                                   verbose=True, 
                                   delta=.00001)
    
    for epoch in range(1, n_epochs + 1):
        for x, y_true in dataloader:
            if torch.cuda.is_available():
                x = x.to('cuda')
                y_true = y_true.to('cuda')
                model = model.to('cuda')

            y_pred = model(x) # forwards pass
            loss_train = loss_fn(y_pred, y_true) # calculate loss
            optimiser.zero_grad() # set gradients to zero
            loss_train.backward() # backwards pass
            optimiser.step() # update model parameters
            train_losses.append(loss_train.item())
        
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        train_losses = [] # clear for next epoch

        writer.add_scalar("Loss/train", train_loss, epoch) # add to tensorboard
        print(f'Epoch {epoch}: ', end='')
                #   f" Validation loss {loss_val.item():.4f}")

        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    model.load_state_dict(torch.load('checkpoint.pt'))

data = DroneMFCCAffectPeak()
dataloader = DataLoader(data, batch_size=16, drop_last=True)
model = SimpleDenseLSTM(20, 124, 2, 3)
optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.MSELoss()

# run the training
training_loop(100, optimiser, model, loss_fn, dataloader)