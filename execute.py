import os
import sys
import time
import datetime
import math
import numpy as np
import torch 
import neptune
import yaml
from utils import modeling
from utils import metrics
from utils import optimizers
from utils import tools


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
            val_batch,
            val_split,  # How much to hold out for validation
            model_type,
            input_size,
            hidden_size,
            output_size,
            metric,
            normalize_input,
            lr,
            bptt,
            epochs,
            scheduler,
            optimizer,
            clip_grad_norm,
            start_trim,
            log_interval,
            val_interval,
            device):
        """Store the info."""
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.device = device
        self.data_size = data_size
        self.val_split = val_split
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.metric = metric
        self.normalize_input = normalize_input
        self.lr = lr
        self.bptt = bptt
        self.epochs = epochs
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        self.start_trim = start_trim
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.min_train_loss = []
        self.max_train_loss = []
        self.val_loss = []
        self.train_loss = []


def batchify(data, bsz, random=False):
    """Divide the dataset into bsz parts."""
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    # from matplotlib import pyplot as plt
    # plt.plot(data[0].cpu());plt.show()
    data = data.view(nbatch, bsz, data.size(1), data.size(2)).contiguous()
    # plt.plot(data[0, 0].cpu());plt.show()
    if type(random) == bool and random == True:
        random = np.random.permutation(len(data))
        data = data[random]
    elif type(random) == np.ndarray:
        data = data[random]
    return data, random  # .to(device)


def get_batch(X, Y, bptt, shared_offset=True):
    """Select from the time axis. Shared_offset is faster."""
    high = X.size(2) - bptt
    assert high > 0
    if shared_offset and bptt > 0:
        offset = np.random.randint(low=0, high=high)
        X = X.narrow(2, offset, bptt)
        Y = Y.narrow(2, offset, bptt)
    elif bptt > 0:
        raise NotImplementedError("Havent worked out narrow with an array yet.")
        offset = np.random.randint(low=0, high=high, size=X.size(1))
        X = X.narrow(2, offset, bptt)
    return X, Y


def train(model, X, Y, optimizer, criterion, scheduler, meta):
    """Loop for training."""
    model.train()
    total_loss = 0.
    start_time = time.time()
    min_loss, max_loss = np.inf, -np.inf
    offset_X, offset_Y = get_batch(X=X, Y=Y, bptt=meta.bptt)
    # for batch, i in enumerate(range(0, X.size(0) - 1, meta.bptt)):
    for batch, (x, y) in enumerate(zip(offset_X, offset_Y)):
        optimizer.zero_grad()
        flattened_y = y.reshape(y.size(0), -1)
        if meta.model_type == "LSTM" or meta.model_type == "GRU":
            output, hiddens = model.forward(x)
        else:
            output = model.forward(x)
        loss = criterion(output.reshape(x.size(0), -1), flattened_y)
        loss.backward()
        if meta.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), meta.clip_grad_norm)
        optimizer.step()
        loss = loss.item()
        total_loss += loss
        min_loss = min(loss, min_loss)
        max_loss = max(loss, max_loss)
        if batch % meta.log_interval == 0 and batch > 0 or batch == (len(offset_X) - 1):
            if scheduler is not None:
                current_lr = scheduler.get_lr()[0]
            else:
                current_lr = meta.lr
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(
                    meta.epoch,
                    batch,
                    len(X) // meta.bptt,
                    current_lr,  # scheduler.get_lr()[0],
                    elapsed * 1000 / meta.log_interval,
                    loss))
            total_loss = 0
            start_time = time.time()
    return min_loss, max_loss


def evaluate(model, X, Y, criterion, meta):
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        offset_X, offset_Y = get_batch(X=X, Y=Y, bptt=meta.bptt)
        for batch, (x, y) in enumerate(zip(offset_X, offset_Y)):
            flattened_y = y.reshape(y.size(0), -1)
            if meta.model_type == "LSTM" or meta.model_type == "GRU":
                output, hiddens = model.forward(x)
            else:
                output = model.forward(x)
            total_loss += criterion(
                output.reshape(x.size(0), -1), flattened_y)
    return total_loss / len(offset_X)


def run_training(
        experiment_name,
        epochs=1000,
        train_batch=49458,
        val_batch=12365,
        dtype=np.float32,
        val_split=0.2,
        shuffle_data=False,
        model_type="GRU",  # 'transformer',
        bptt=600,
        hidden_size=3,
        lr=1e-1,
        start_trim=600,
        log_interval=5,
        val_interval=20,
        clip_grad_norm=False,
        output_dir='results',
        normalize_input=True,
        optimizer='Adam',
        scheduler=None,  # 'StepLR',
        metric='pearson'):  # pearson
    """Run training and validation."""
    if USE_NEPTUNE:
        neptune.init("Serre-Lab/deepspine")
        if experiment_name is None:
            experiment_name = "synthetic_data"
        neptune.create_experiment(experiment_name)
    default_model_params = tools.get_model_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    data = np.load(DATA_FILE)
    mn = data["mn"]
    ees = data["ees"]
    kinematics = data["kinematics"]
    X = torch.from_numpy(np.concatenate((ees, kinematics), 1).astype(dtype))
    Y = torch.from_numpy(mn.astype(dtype))
    X = X.permute(0, 2, 1)
    Y = Y.permute(0, 2, 1)
    input_size = X.size(-1)
    output_size = Y.size(-1)
    meta = Meta(
        data_size=X.shape,
        train_batch=train_batch,
        val_batch=val_batch,
        val_split=val_split,
        model_type=model_type,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        metric=metric,
        normalize_input=normalize_input,
        lr=lr,
        bptt=bptt,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_norm=clip_grad_norm,
        log_interval=log_interval,
        val_interval=val_interval,
        start_trim=start_trim,
        device=device)

    # Prepare data
    if meta.start_trim:
        X = X.narrow(1, meta.start_trim, X.size(1) - meta.start_trim)
        Y = Y.narrow(1, meta.start_trim, Y.size(1) - meta.start_trim)

    if shuffle_data:
        idx = np.random.permutation(len(X))
        X = X[idx]
        Y = Y[idx]

    if meta.normalize_input:
        # X = (X - 127.5) / 127.5
        # Y = (Y - 127.5) / 127.5
        X = (X - X.mean(1).unsqueeze(1)) / (X.std(1).unsqueeze(1) + 1e-4)  # This is peaking but whatever...
        Y = (Y - Y.mean(1).unsqueeze(1)) / (Y.std(1).unsqueeze(1) + 1e-4)
    X = X.to(meta.device)
    Y = Y.to(meta.device)
    cv_idx = np.arange(len(X))
    cv_idx = cv_idx > np.round(float(len(X)) * val_split).astype(int)
    X_train = X[cv_idx]
    Y_train = Y[cv_idx]
    X_val = X[~cv_idx]
    Y_val = Y[~cv_idx]
    assert meta.train_batch < len(X_train), "Train batch size > dataset size {}.".format(len(X_train))
    assert meta.val_batch < len(X_val), "Val batch size > dataset size {}.".format(len(X_val))

    # Create model
    model = modeling.create_model(
        model_type=meta.model_type,
        input_size=meta.input_size,
        hidden_size=meta.hidden_size,
        output_size=meta.output_size,
        default_model_params=default_model_params,
        device=meta.device)
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
    X_val, _ = batchify(X_val, bsz=meta.val_batch, random=False)
    Y_val, _ = batchify(Y_val, bsz=meta.val_batch, random=False)
    for epoch in range(1, meta.epochs + 1):
        epoch_start_time = time.time()
        meta.epoch = epoch
        X_train_i, random_idx = batchify(X_train, bsz=meta.train_batch, random=True)
        Y_train_i, _ = batchify(Y_train, bsz=meta.train_batch, random=random_idx)
        min_train_loss, max_train_loss = train(
            model=model,
            X=X_train_i,
            Y=Y_train_i,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            meta=meta)
        if epoch % meta.val_interval == 0:
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
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'.format(
                  epoch,
                  (time.time() - epoch_start_time),
                  meta.val_loss[-1]))
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
    _, experiment_name = sys.argv
    run_training(experiment_name)

