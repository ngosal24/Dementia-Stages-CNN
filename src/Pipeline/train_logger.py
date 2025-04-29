import json
import os

class TrainingLogger:
    def __init__(self, save_path):
        self.save_path = save_path
        self.epoch_data = []

    def log(self, epoch, loss, lr):
        self.epoch_data.append({
            'epoch': epoch,
            'loss': loss,
            'lr': lr
        })

    def save(self):
        dir_path = os.path.dirname(self.save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(self.epoch_data, f, indent=4)
