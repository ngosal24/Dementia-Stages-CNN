#config.py
import os
import torch

class Config:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # data_dir = os.path.join(project_root, "Dataset4")
    data_dir = os.path.join(project_root, "Dataset3")

    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Separate paths for clarity
    best_model_path = os.path.join(project_root, "best_model.pth")
    final_model_path = os.path.join(project_root, "final_model.pth")

    last_model_path = os.path.join(project_root, "last_model.pth")
    score_log_path = os.path.join(project_root, "model_scores.json")
