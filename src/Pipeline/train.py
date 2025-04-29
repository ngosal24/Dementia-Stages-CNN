import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from .model import get_model
from .data_loader import get_dataloaders_with_validation, get_labels_from_subset
from .utils import save_model
from .config import Config
from .train_logger import TrainingLogger


def train(model, train_loader, val_loader, config, use_focal=False, resume=False):
    model.to(config.device)

    labels = get_labels_from_subset(train_loader.dataset)
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(labels),
                                         y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_f1 = 0.0
    best_epoch = 0
    logger = TrainingLogger("training_log.json")

    for epoch in range(config.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            images, labels = images.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation F1-score
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.device), labels.to(config.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"Validation F1-score: {f1:.4f}")

        # Save best model based on F1
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            save_model(model, path=config.best_model_path)
            print(f"✅ New best model saved with F1-score: {f1:.4f}")

        logger.log(epoch + 1, avg_loss, optimizer.param_groups[0]['lr'])
        scheduler.step(avg_loss)

    # Save last model after training ends
    save_model(model, path=config.last_model_path)
    print("📦 Last model saved.")

    # Save scores to JSON
    scores = {
        "best_model": {
            "f1_weighted": best_f1,
            "epoch": best_epoch,
            "path": config.best_model_path
        },
        "last_model": {
            "f1_weighted": f1,
            "epoch": config.epochs,
            "path": config.last_model_path
        },
        "final_model": {
            "path": config.final_model_path,
            "source": None,  # will be updated by user
            "timestamp": datetime.now().isoformat()
        }
    }

    with open(config.score_log_path, 'w') as f:
        json.dump(scores, f, indent=4)

    logger.save()


def train_model(resume=False):
    config = Config()
    train_loader, val_loader, test_loader, class_names, _ = get_dataloaders_with_validation(
        config.data_dir,
        config.batch_size,
        val_split=0.2
    )

    model = get_model(num_classes=len(class_names))

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        use_focal=False,
        resume=resume
    )

    return model, test_loader, class_names
