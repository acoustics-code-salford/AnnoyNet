import torch
import glob, os
import numpy as np
import pandas as pd
import torchaudio, torchvision

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    '''
    Adapted from https://github.com/Bjarten/early-stopping-pytorch.
    Early stops the training if validation loss doesn't improve
    after a given patience.
    '''
    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path='checkpoint.pt',
        trace_func=print
    ):
        '''
        Args:
            patience (int): Epochs to wait after last time val loss improved.
                            Default: 7
            verbose (bool): Prints message for each val loss improvement.
                            Default: False
            delta (float):  Min change in the counting as improvement.
                            Default: 0
            path (str):     Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): Trace print function.
                            Default: print
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                'EarlyStopping counter: ',
                f'{self.counter} out of {self.patience}',
                end='\r'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                'Validation loss decreased ',
                f'({self.val_loss_min:.6f} --> {val_loss:.6f}). ',
                'Saving model ...',
                end='\r'
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def training_loop(n_epochs,
                  optimiser,
                  model,
                  loss_fn,
                  dataloader,
                  patience=20,
                  writer=SummaryWriter()):

    train_losses = []
    avg_train_losses = []
    early_stopping = EarlyStopping(patience=patience,
                                   verbose=True,
                                   delta=.00001)
    
    model.train()

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

# data = DroneMFCCAffectPeak()
# dataloader = DataLoader(data, batch_size=16, drop_last=True)
# model = SimpleDenseLSTM(20, 124, 2, 3)
# optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# loss_fn = nn.MSELoss()

# # run the training
# training_loop(100, optimiser, model, loss_fn, dataloader)
