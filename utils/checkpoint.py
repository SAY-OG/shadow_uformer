import torch
import os

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer': optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler': scheduler.load_state_dict(checkpoint['scheduler'])

    if scaler and 'scaler' in checkpoint: scaler.load_state_dict(checkpoint['scaler'])
    else:
        print("No scheduler state found. Training is starting with initial learning rate")

    return checkpoint.get('epoch', 0)
