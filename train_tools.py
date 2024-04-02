import torch
import numpy as np
import matplotlib.pyplot as plt


class EarlyStopping:
    '''
    Adapted from https://github.com/Bjarten/early-stopping-pytorch
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


def train_model(
        n_epochs,
        optimiser,
        model,
        loss_fn,
        train_dataloader,
        val_dataloader,
        patience=20,
        plot=True
):

    train_losses = []
    val_losses = []
    avg_train_losses = []
    avg_val_losses = []
    early_stopping = EarlyStopping(patience=patience,
                                   verbose=True,
                                   delta=.00001)
    
    if torch.cuda.is_available():
        model = model.to('cuda')

    for epoch in range(1, n_epochs + 1):
        
        model.train()
        for x, y_true in train_dataloader:
            if torch.cuda.is_available():
                x = x.to('cuda')
                y_true = y_true.to('cuda')

            y_pred = model(x).squeeze()  # forwards pass
            loss = loss_fn(y_pred, y_true)  # calculate loss
            optimiser.zero_grad()  # set gradients to zero
            loss.backward()  # backwards pass
            optimiser.step()  # update model parameters
            train_losses.append(loss.item())
        
        model.eval()
        for x, y_true in val_dataloader:
            
            if torch.cuda.is_available():
                x = x.to('cuda')
                y_true = y_true.to('cuda')

            y_pred = model(x).squeeze() # forwards pass
            val_loss = loss_fn(y_pred, y_true) # calculate loss
            val_losses.append(val_loss.item())

        train_loss = np.average(train_losses)
        val_loss = np.average(val_losses)
        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)
        train_losses = []  # clear for next epoch
        val_losses = []

        print(f'Epoch {epoch}: ', end='')
        last_improved = epoch
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            last_improved -= patience
            break
    
    model.load_state_dict(torch.load('checkpoint.pt'))

    if plot:
        epochs = range(1, epoch+1)
        plt.plot(epochs, avg_train_losses, label='Train')
        plt.plot(epochs, avg_val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.axvline(last_improved, color='r', ls='--', lw=1)
        plt.legend()


def test_model(model, test_dataloader, loss_fn, metric):

    model.eval()

    loss_aggr = 0.0
    metric_scores = torch.tensor(())

    with torch.no_grad():
        for x, y_true in test_dataloader:
            if torch.cuda.is_available():
                x = x.to('cuda')
                y_true = y_true.to('cuda')
                model = model.to('cuda')
                metric_scores = metric_scores.to('cuda')

            y_pred = model(x)
            loss = loss_fn(y_pred, y_true.unsqueeze(1))
            loss_aggr += loss.item()

            score = metric(y_pred, y_true).mean()
            metric_scores = torch.cat((metric_scores, score.reshape(1)))

    loss_aggr /= len(test_dataloader)
    metric_aggr = metric_scores.sum() / len(test_dataloader)

    return metric_aggr.item(), loss_aggr
