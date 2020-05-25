import datetime
import time
import os
import numpy as np
import torch 
import neptune
from models.create_model import create_model
from utils import metrics
from utils import optimizers


np.random.seed(0)
torch.manual_seed(0)
DATA_DIR  = "data"
DATA_FILE = os.path.join(DATA_DIR, "processed_data_uint8.npz")
USE_NEPTUNE = True


class Meta:
    """Hold experiment meta info."""
    def __init__(
            self,
            data_size,
            train_batch,
            test_batch,
            val_split,  # How much to hold out for validation
            model_type,
            metric,
            lr,
            bptt,
            epochs,
            scheduler,
            optimizer,
            device):
        """Store the info."""
        self.train_batch = train_batch
        self.tesr_batch = test_batch
        self.device = device
        self.data_size = data_size
        self.val_split = val_split
        self.model_type = model_type
        self.metric = metric
        self.lr = lr
        self.bptt = bptt
        self.epochs = epochs
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.val_loss = []
        self.train_loss = []


def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data  # .to(device)


def get_batch(X, Y, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i: i + seq_len]
    target = source[i + 1: i + 1 + seq_len].view(-1)
    return data, target


def get_batch(X, Y, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i: i + seq_len]
    target = source[i + 1: i + 1 + seq_len].view(-1)
    return data, target


def train(model, X, Y, optimizer, criterion, meta):
    """Loop for training."""
    model.train()
    total_loss = 0.
    start_time = time.time()
    min_loss, max_loss = np.inf, -np.inf
    for batch, i in enumerate(range(0, X.size(0) - 1, meta.bptt)):
        x, y = get_batch(X=X, Y=Y, i=i)
        optimizer.zero_grad()
        if meta.model_type == "LSTM":
            output, _ = model(x, (torch.randn(1, 1, 3), torch.randn(1, 1, 3)))
        elif meta.model_type == "GRU":
            output, _ = model(x, (torch.randn(1, 1, 3)))
        else:
            output = model(x)
        loss = criterion(output, targets)
        loss.backward()
        if meta.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), meta.clip_grad_norm)
        optimizer.step()
        loss = loss.item()
        total_loss += loss
        min_loss = min(loss, min_loss)
        max_loss = max(loss, max_loss)
        if batch % meta.log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    meta.epoch,
                    batch,
                    len(X) // bptt,
                    scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss,
                    math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return min_loss, max_loss


def evaluate(model, X, Y, criterion, meta):
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, X.size(0) - 1, bptt):
            x, y = get_batch(X=X, Y=Y, i=i)
            output = model(x)
            total_loss += len(x) * criterion(output, targets).item()
    return total_loss / (len(X) - 1)


def run_training(
        epochs=100,
        train_batch=128,
        test_batch=128,
        dtype=np.float32,
        val_split=0.2,
        shuffle_data=True,
        model_type='transformer',
        bptt=400,
        lr=1e-2,
        output_dir='results',
        optimizer='AdamW',
        scheduler='StepLR',
        metric='pearson'):
    """Run training and validation."""
    if USE_NEPTUNE:
        neptune.init("Serre-Lab/deepspine")
        neptune.create_experiment("synthetic_data_v0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    data = np.load(DATA_FILE)
    mn = data["mn"]
    ees = data["ees"]
    kinematics = data["kinematics"]
    X = torch.from_numpy(np.concatenate((ees, kinematics), 1).astype(dtype))
    Y = torch.from_numpy(mn.astype(dtype))
    meta = Meta(
        data_size=X.shape,
        train_batch=train_batch,
        test_batch=test_batch,
        val_split=val_split,
        model_type=model_type,
        metric=metric,
        lr=lr,
        bptt=bptt,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device)

    # Prepare data
    if shuffle_data:
        idx = np.random.permutation(len(X))
        X = X[idx]
        Y = Y[idx]
    X = X.to(meta.device)
    Y = Y.to(meta.device)
    cv_idx = np.arange(len(X))
    cv_idx = cv_idx > np.round(float(len(X)) * val_split).astype(int)
    X_train = X[cv_idx]
    Y_train = Y[cv_idx]
    X_val = X[~cv_idx]
    Y_val = Y[~cv_idx]
    X_train = batchify(X_train, bsz=meta.train_batch)
    Y_train = batchify(Y_train, bsz=meta.train_batch)
    X_val = batchify(X_val, bsz=meta.val_batch)
    Y_val = batchify(Y_val, bsz=meta.val_batch)

    # Create model
    model = create_model(
        model_type=meta.model_type,
        ntoken=255,
        ninp=2,
        nhead=2,
        nhid=2,
        nlayers=1,
        dropout=0.5).to(meta.device)
    criterion = metrics.get_metric(metric)
    optimizer = optimizers.get_optimizer(optimizer)
    assert lr < 1, "LR is greater than 1."
    optimizer = optimizer(model.parameters(), lr=lr)
    if scheduler is not None:
        scheduler = optimizers.get_scheduler(scheduler) 
        scheduler = scheduler(optimizer)

    # Start training
    best_val_loss = float("inf")
    best_model = None
    for epoch in range(1, meta.epochs + 1):
        epoch_start_time = time.time()
        min_train_loss, max_train_loss = train(
            model=model,
            X=X_train,
            Y=Y_train,
            optimizer=optimizer,
            criterion=criterion,
            meta=meta)
        val_loss = evaluate(
            model=model,
            X=X_val,
            Y=Y_val,
            criterion=criterion,
            meta=meta)
        meta.min_train_loss.append(min_train_loss)
        meta.max_train_loss.append(max_train_loss)
        meta.val_loss.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(
                  epoch,
                  (time.time() - epoch_start_time),
                  meta.val_loss,
                  math.exp(meta.val_loss)))
        print('-' * 89)
        if USE_NEPTUNE:
            neptune.log_metric('min_train_loss', min_train_loss)
            neptune.log_metric('max_train_loss', max_train_loss)
            neptune.log_metric('val_loss', val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
        if scheduler is not None:
            scheduler.step()
    np.savez(os.path.join(output_dir, 'results_{}'.format(timestamp)), **meta.__dict__)


if __name__ == '__main__':
    run_training()

