import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df, input_cols, target_col):
    X = df[input_cols].values
    y = df[target_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

class TrafficDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def prepare_dataloaders(data_path, input_cols, target_col, test_size=0.2, batch_size=32):
    df = load_dataset(data_path)
    X, y, scaler = preprocess_data(df, input_cols, target_col)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    train_dataset = TrafficDataset(X_train, y_train)
    val_dataset = TrafficDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler

if __name__ == '__main__':
    data_path = 'data/pems08.csv'
    input_columns = ['speed', 'volume', 'occupancy']
    target_column = 'flow'
    train_loader, val_loader, scaler = prepare_dataloaders(data_path, input_columns, target_column)
    for batch in train_loader:
        x, y = batch
        print(x.shape, y.shape)
        break
