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
            score,
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
            train_weight,
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
        self.score = score
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
        self.train_weight = train_weight
        self.min_train_loss = []
        self.max_train_loss = []
        self.val_loss = []
        self.val_score = []
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


def train(model, X, Y, optimizer, criterion, score, scheduler, meta):
    """Loop for training."""
    model.train()
    total_loss = 0.
    start_time = time.time()
    min_loss, max_loss = np.inf, -np.inf
    offset_X, offset_Y = get_batch(X=X, Y=Y, bptt=meta.bptt)
    # for batch, i in enumerate(range(0, X.size(0) - 1, meta.bptt)):
    for batch, (x, y) in enumerate(zip(offset_X, offset_Y)):
        optimizer.zero_grad()
        if meta.model_type == "LSTM" or meta.model_type == "GRU":
            output, hiddens = model.forward(x)
        else:
            output = model.forward(x)
        if meta.metric == "bce":
            loss = criterion(
                pos_weight=torch.ones(output.size(2)).to(meta.device) * meta.train_weight)(output, y)
        else:
            loss = criterion(output, y)
        loss.backward()
        if meta.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), meta.clip_grad_norm)
        optimizer.step()
        loss = loss.item()
        total_loss += loss
        min_loss = min(loss, min_loss)
        max_loss = max(loss, max_loss)
        if batch % meta.log_interval == 0 and batch > 0 or batch == (len(offset_X) - 1):
            sc = score(output, y)
            if scheduler is not None:
                current_lr = scheduler.get_lr()[0]
            else:
                current_lr = meta.lr
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | score {:5.2f}'.format(
                    meta.epoch,
                    batch,
                    len(X) // meta.bptt,
                    current_lr,  # scheduler.get_lr()[0],
                    elapsed * 1000 / meta.log_interval,
                    loss,
                    sc))
            total_loss = 0
            start_time = time.time()
    return min_loss, max_loss, output, y


def evaluate(model, X, Y, criterion, score, meta):
    """Score the validation set."""
    model.eval() # Turn on the evaluation mode
    total_loss, total_sc = 0., 0.
    with torch.no_grad():
        offset_X, offset_Y = get_batch(X=X, Y=Y, bptt=meta.bptt)
        for batch, (x, y) in enumerate(zip(offset_X, offset_Y)):
            # flattened_y = y.reshape(y.size(0), -1)
            if meta.model_type == "LSTM" or meta.model_type == "GRU":
                output, hiddens = model.forward(x)
            else:
                output = model.forward(x)
            if meta.metric == "bce":
                total_loss += criterion()(output, y)
            else:
                total_loss += criterion(output, y)
            total_sc += score(output, y)
    return total_loss / len(offset_X), total_sc / len(offset_X), output, y


def run_training(
        experiment_name,
        debug=False,
        epochs=10000,
        train_batch=58732 // 2,
        val_batch=3091,
        dtype=np.float32,
        val_split=0.05,
        shuffle_data=True,
        model_type="GRU",  # 'transformer',
        bptt=700,
        hidden_size=3,
        lr=1e-2,
        start_trim=650,
        log_interval=5,
        val_interval=20,
        clip_grad_norm=False,
        output_dir="results",
        normalize_input=True,
        optimizer="AdamW",
        scheduler="StepLR",
        train_weight=10.,
        score="pearson",
        metric="l2_pearson"):  # pearson
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
        score=score,
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
        train_weight=train_weight,
        device=device)

    # Prepare data
    if meta.start_trim:
        X = X.narrow(1, meta.start_trim, X.size(1) - meta.start_trim)
        Y = Y.narrow(1, meta.start_trim, Y.size(1) - meta.start_trim)

    if shuffle_data:
        idx = np.random.permutation(len(X))
        X = X[idx]
        Y = Y[idx]

    if meta.metric == "bce":
        # Quantize Y
        Y = (Y > 127.5).float()

    if meta.normalize_input:
        # X = (X - 127.5) / 127.5
        # Y = (Y - 127.5) / 127.5
        X = (X - X.mean(1, keepdim=True)) / (X.std(1, keepdim=True) + 1e-8)  # This is peaking but whatever...
        if meta.metric != "bce":
            Y = (Y - Y.mean(1, keepdim=True)) / (Y.std(1, keepdim=True) + 1e-8)
    X = X.to(meta.device)
    Y = Y.to(meta.device)
    cv_idx = np.arange(len(X))
    cv_idx = cv_idx > np.round(float(len(X)) * val_split).astype(int)
    X_train = X[cv_idx]
    Y_train = Y[cv_idx]
    X_val = X[~cv_idx]
    Y_val = Y[~cv_idx]
    assert meta.train_batch < len(X_train), "Train batch size > dataset size {}.".format(len(X_train) - 1)
    assert meta.val_batch < len(X_val), "Val batch size > dataset size {}.".format(len(X_val) - 1)

    # Create model
    model = modeling.create_model(
        model_type=meta.model_type,
        input_size=meta.input_size,
        hidden_size=meta.hidden_size,
        output_size=meta.output_size,
        default_model_params=default_model_params,
        device=meta.device)
    score, criterion = metrics.get_metric(metric)
    optimizer_fun = optimizers.get_optimizer(optimizer)
    assert lr < 1, "LR is greater than 1."
    if "adam" in optimizer.lower():
        optimizer = optimizer_fun(model.parameters(), lr=lr, amsgrad=True)
    else:
        optimizer = optimizer_fun(model.parameters(), lr=lr)
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
        min_train_loss, max_train_loss, train_output, train_gt = train(
            model=model,
            X=X_train_i,
            Y=Y_train_i,
            optimizer=optimizer,
            criterion=criterion,
            score=score,
            scheduler=scheduler,
            meta=meta)
        if epoch % meta.val_interval == 0:
            val_loss, val_score, val_output, val_gt = evaluate(
                model=model,
                X=X_val,
                Y=Y_val,
                criterion=criterion,
                score=score,
                meta=meta)
            meta.min_train_loss.append(min_train_loss)
            meta.max_train_loss.append(max_train_loss)
            meta.val_loss.append(val_loss)
            meta.val_score.append(val_score)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid score {:5.2f}'.format(
                  epoch,
                  (time.time() - epoch_start_time),
                  meta.val_loss[-1],
                  meta.val_score[-1]))
            print('-' * 89)
            if USE_NEPTUNE:
                neptune.log_metric('min_train_loss', min_train_loss)
                neptune.log_metric('max_train_loss', max_train_loss)
                neptune.log_metric('val_{}'.format(meta.metric), val_loss)
                neptune.log_metric('val_pearson', val_score)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
            if val_loss < 0.65 and debug:
                from matplotlib import pyplot as plt
                fig = plt.figure()
                plt.title('val')
                plt.subplot(211)
                plt.plot(val_output[50].cpu())
                plt.subplot(212)
                plt.plot(val_gt[50].cpu())
                plt.show()
                plt.close(fig)
                fig = plt.figure()
                plt.title('train')
                plt.subplot(211)
                plt.plot(train_output[50].cpu().detach())
                plt.subplot(212)
                plt.plot(train_gt[50].cpu())
                plt.show()
                plt.close(fig)
            if scheduler is not None:
                scheduler.step()

    # Fix some type issues
    meta.val_loss = [x.cpu() for x in meta.val_loss]
    meta.val_score = [x.cpu() for x in meta.val_score]
    meta.min_train_loss = [x.cpu() for x in meta.min_train_loss]
    meta.max_train_loss = [x.cpu() for x in meta.max_train_loss]
    np.savez(os.path.join(output_dir, '{}results_{}'.format(experiment_name, timestamp)), **meta.__dict__)  # noqa


if __name__ == '__main__':
    _, experiment_name = sys.argv
    run_training(experiment_name)

