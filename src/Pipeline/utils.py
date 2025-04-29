#utils.py

import torch
from .config import Config

def save_model(model, path='best_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='best_model.pth'):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


def get_config():
    return Config()
