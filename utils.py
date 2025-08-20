import os
import numpy as np
import torch
import random
import logging
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_logger(log_dir, log_name='log.txt'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, log_name))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normalize_data(data, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    return data, scaler

def denormalize_data(data, scaler):
    return scaler.inverse_transform(data)

def generate_batches(data, batch_size, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len:i+seq_len+pred_len]
        X.append(x)
        Y.append(y)
    X = np.stack(X)
    Y = np.stack(Y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def load_csv_file(path):
    df = pd.read_csv(path)
    return df

def split_train_val_test(data, train_ratio=0.6, val_ratio=0.2):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def calculate_metrics(y_true, y_pred):
    mae = F.l1_loss(y_pred, y_true).item()
    mse = F.mse_loss(y_pred, y_true).item()
    rmse = np.sqrt(mse)
    mape = (torch.abs((y_true - y_pred) / y_true).mean() * 100).item()
    return mae, mape, rmse

def timing(f):
    def wrap(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f'Function {f.__name__} took {(end-start):.3f}s')
        return result
    return wrap

def set_device(model, device):
    return model.to(device)

def shuffle_data(X, Y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], Y[indices]