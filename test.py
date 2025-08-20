import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from model import OTSTNet
from data import TrafficDataset
from utils import MAE, MAPE, RMSE, load_config
import itertools
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = "config.yaml"
config = load_config(config_path)

test_data = TrafficDataset(config['data']['test_path'],
                           window_size=config['data']['window_size'],
                           horizon=config['data']['horizon'],
                           normalize=True)

test_loader = DataLoader(test_data, batch_size=config['train']['batch_size'], shuffle=False)

model = OTSTNet(node_features=config['model']['node_features'],
                hidden_dim=config['model']['hidden_dim'],
                temporal_dim=config['model']['temporal_dim'],
                num_heads=config['model']['num_heads'],
                num_layers=config['model']['num_layers']).to(device)

checkpoint = torch.load(config['train']['checkpoint_path'], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

mae_list = []
rmse_list = []
mape_list = []

all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        mae_list.append(MAE(pred, y).item())
        rmse_list.append(RMSE(pred, y).item())
        mape_list.append(MAPE(pred, y).item())

mean_mae = np.mean(mae_list)
mean_rmse = np.mean(rmse_list)
mean_mape = np.mean(mape_list)

print("Test MAE:", mean_mae)
print("Test RMSE:", mean_rmse)
print("Test MAPE:", mean_mape)

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

def plot_predictions(preds, targets, step=0):
    plt.figure(figsize=(12,6))
    plt.plot(targets[step,:,0], label='Ground Truth')
    plt.plot(preds[step,:,0], label='Prediction')
    plt.title(f"Traffic Flow Prediction at Sample {step}")
    plt.xlabel("Node")
    plt.ylabel("Traffic Value")
    plt.legend()
    plt.show()

plot_predictions(all_preds, all_targets, step=0)
plot_predictions(all_preds, all_targets, step=1)

horizon_preds = all_preds[:,:,-config['data']['horizon']:]
horizon_targets = all_targets[:,:,-config['data']['horizon']:]
for t in range(config['data']['horizon']):
    plt.figure(figsize=(12,6))
    plt.plot(horizon_targets[:,t,0], label=f'Target t+{t+1}')
    plt.plot(horizon_preds[:,t,0], label=f'Prediction t+{t+1}')
    plt.title(f"Multi-step Traffic Prediction t+{t+1}")
    plt.xlabel("Node")
    plt.ylabel("Traffic Value")
    plt.legend()
    plt.show()

dk_values = [32, 64, 128, 256, 512]
dh_values = [32, 64, 128, 256, 512]
H_values = [1, 2, 4, 8]

sensitivity_results = []

for dk, dh, H in itertools.product(dk_values, dh_values, H_values):
    model = OTSTNet(node_features=config['model']['node_features'],
                    hidden_dim=dh,
                    temporal_dim=dh,
                    num_heads=H,
                    num_layers=config['model']['num_layers']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    mae_temp = []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            mae_temp.append(MAE(pred, y).item())
    mean_mae_temp = np.mean(mae_temp)
    sensitivity_results.append((dk, dh, H, mean_mae_temp))

import pandas as pd
df_sensitivity = pd.DataFrame(sensitivity_results, columns=['d_k','d_h','H','MAE'])
df_sensitivity.to_csv("sensitivity_results.csv", index=False)
print(df_sensitivity)

fig = plt.figure(figsize=(12,8))
for H in H_values:
    subset = df_sensitivity[df_sensitivity['H']==H]
    plt.plot(subset['d_k'], subset['MAE'], label=f'H={H}')
plt.xlabel('d_k')
plt.ylabel('MAE')
plt.title('Hyperparameter Sensitivity Analysis')
plt.legend()
plt.show()

for dh in dh_values:
    subset = df_sensitivity[df_sensitivity['d_h']==dh]
    plt.plot(subset['H'], subset['MAE'], label=f'd_h={dh}')
plt.xlabel('H')
plt.ylabel('MAE')
plt.title('Hyperparameter Sensitivity Analysis by d_h')
plt.legend()
plt.show()

for dk in dk_values:
    subset = df_sensitivity[df_sensitivity['d_k']==dk]
    plt.plot(subset['H'], subset['MAE'], label=f'd_k={dk}')
plt.xlabel('H')
plt.ylabel('MAE')
plt.title('Hyperparameter Sensitivity Analysis by d_k')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(all_targets[:50,0,0], label='Ground Truth')
plt.plot(all_preds[:50,0,0], label='Prediction')
plt.title("First 50 Nodes Single-step Prediction")
plt.xlabel("Node Index")
plt.ylabel("Traffic Value")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(all_targets[0,:50,0], label='Ground Truth')
plt.plot(all_preds[0,:50,0], label='Prediction')
plt.title("First 50 Time Steps Prediction")
plt.xlabel("Time Step")
plt.ylabel("Traffic Value")
plt.legend()
plt.show()

def plot_node_traffic(node_idx):
    plt.figure(figsize=(12,6))
    plt.plot(all_targets[:,node_idx,0], label='Target')
    plt.plot(all_preds[:,node_idx,0], label='Pred')
    plt.title(f'Traffic Flow at Node {node_idx}')
    plt.xlabel('Time Step')
    plt.ylabel('Traffic Value')
    plt.legend()
    plt.show()

for node in range(3):
    plot_node_traffic(node)

def plot_sample_multistep(sample_idx):
    plt.figure(figsize=(12,6))
    plt.plot(horizon_targets[sample_idx,:,0], label='Target Horizon')
    plt.plot(horizon_preds[sample_idx,:,0], label='Prediction Horizon')
    plt.title(f'Multi-step Prediction Sample {sample_idx}')
    plt.xlabel('Horizon Step')
    plt.ylabel('Traffic Value')
    plt.legend()
    plt.show()

for sample in range(3):
    plot_sample_multistep(sample)

df_sensitivity['d_k_dh_H'] = df_sensitivity.apply(lambda row: f"{row['d_k']}_{row['d_h']}_{row['H']}", axis=1)
plt.figure(figsize=(14,8))
for key, grp in df_sensitivity.groupby('d_k_dh_H'):
    plt.plot(grp['H'], grp['MAE'], label=key)
plt.xlabel('H')
plt.ylabel('MAE')
plt.title('Combined Hyperparameter Sensitivity')
plt.legend()
plt.show()

plt.figure(figsize=(14,8))
for dk in dk_values:
    subset = df_sensitivity[df_sensitivity['d_k']==dk]
    plt.plot(subset['H'], subset['MAE'], label=f'd_k={dk}')
plt.xlabel('H')
plt.ylabel('MAE')
plt.title('d_k vs H Sensitivity')
plt.legend()
plt.show()

plt.figure(figsize=(14,8))
for dh in dh_values:
    subset = df_sensitivity[df_sensitivity['d_h']==dh]
    plt.plot(subset['H'], subset['MAE'], label=f'd_h={dh}')
plt.xlabel('H')
plt.ylabel('MAE')
plt.title('d_h vs H Sensitivity')
plt.legend()
plt.show()

plt.figure(figsize=(14,8))
for H in H_values:
    subset = df_sensitivity[df_sensitivity['H']==H]
    plt.plot(subset['d_k'], subset['MAE'], label=f'H={H}')
plt.xlabel('d_k')
plt.ylabel('MAE')
plt.title('d_k vs MAE for different H')
plt.legend()
plt.show()

plt.figure(figsize=(14,8))
for H in H_values:
    subset = df_sensitivity[df_sensitivity['H']==H]
    plt.plot(subset['d_h'], subset['MAE'], label=f'H={H}')
plt.xlabel('d_h')
plt.ylabel('MAE')
plt.title('d_h vs MAE for different H')
plt.legend()
plt.show()

plt.figure(figsize=(14,8))
for dk in dk_values:
    subset = df_sensitivity[df_sensitivity['d_k']==dk]
    plt.plot(subset['d_h'], subset['MAE'], label=f'd_k={dk}')
plt.xlabel('d_h')
plt.ylabel('MAE')
plt.title('d_h vs MAE for different d_k')
plt.legend()
plt.show()

plt.figure(figsize=(14,8))
plt.plot(df_sensitivity['d_k'], df_sensitivity['MAE'], 'o-')
plt.xlabel('d_k')
plt.ylabel('MAE')
plt.title('Scatter MAE vs d_k')
plt.show()

plt.figure(figsize=(14,8))
plt.plot(df_sensitivity['d_h'], df_sensitivity['MAE'], 'o-')
plt.xlabel('d_h')
plt.ylabel('MAE')
plt.title('Scatter MAE vs d_h')
plt.show()

plt.figure(figsize=(14,8))
plt.plot(df_sensitivity['H'], df_sensitivity['MAE'], 'o-')
plt.xlabel('H')
plt.ylabel('MAE')
plt.title('Scatter MAE vs H')
plt.show()

plt.figure(figsize=(14,8))
plt.plot(df_sensitivity['d_k']+df_sensitivity['d_h'], df_sensitivity['MAE'], 'o-')
plt.xlabel('d_k + d_h')
plt.ylabel('MAE')
plt.title('Combined Dimension vs MAE')
plt.show()

plt.figure(figsize=(14,8))
plt.plot(df_sensitivity['d_k']*df_sensitivity['H'], df_sensitivity['MAE'], 'o-')
plt.xlabel('d_k * H')
plt.ylabel('MAE')
plt.title('d_k * H vs MAE')
plt.show()

plt.figure(figsize=(14,8))
plt.plot(df_sensitivity['d_h']*df_sensitivity['H'], df_sensitivity['MAE'], 'o-')
plt.xlabel('d_h * H')
plt.ylabel('MAE')
plt.title('d_h * H vs MAE')
plt.show()
