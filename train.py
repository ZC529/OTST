import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import OTSTModel
from data import TrafficDataset
from utils import load_config, set_seed, evaluate

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    config = load_config('config.yaml')
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = TrafficDataset(config['data']['train'])
    val_dataset = TrafficDataset(config['data']['val'])
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)
    model = OTSTModel(config).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    best_loss = float('inf')
    for epoch in range(config['train']['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{config['train']['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), config['train']['save_path'])
    model.load_state_dict(torch.load(config['train']['save_path']))
    test_dataset = TrafficDataset(config['data']['test'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()
