import torch
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from Pipeline.config import Config
from Pipeline.model import get_model
from Pipeline.utils import load_model
from Pipeline.data_loader import get_dataloaders_with_validation
from Pipeline.evaluate import evaluate_model
from Pipeline.html_report import generate_html_report


def save_confusion_matrix(cm, class_names, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_f1_bar_plot(report_dict, filename="f1_scores.png"):
    df = pd.DataFrame(report_dict).transpose()
    if "accuracy" in df.index:
        df = df.drop("accuracy")
    df = df[df.index != "macro avg"]
    df = df[df.index != "weighted avg"]
    f1s = df["f1-score"]

    plt.figure(figsize=(8, 5))
    f1s.plot(kind='bar', color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    cfg = Config()
    _, _, test_loader, class_names, _ = get_dataloaders_with_validation(cfg.data_dir, cfg.batch_size)

    model = get_model(num_classes=len(class_names))
    model = load_model(model, path=cfg.best_model_path)
    model.to(cfg.device)
    model.eval()

    # 📂 Save paths relative to this file
    save_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(save_dir, "classification_report.json")
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    f1_plot_path = os.path.join(save_dir, "f1_scores.png")
    html_path = os.path.join(save_dir, "test_report.html")

    max_attempts = 1
    attempts = 0
    achieved_f1 = 0.0

    while achieved_f1 < 0.70 and attempts < max_attempts:
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(cfg.device), labels.to(cfg.device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # 📊 Save classification report
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        achieved_f1 = report_dict['weighted avg']['f1-score']
        print(f"\n🔁 Attempt {attempts + 1} — Avg F1 Score: {achieved_f1:.4f}\n")
        print(json.dumps(report_dict, indent=4))

        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)

        # 🧊 Save confusion matrix image
        cm = confusion_matrix(y_true, y_pred)
        save_confusion_matrix(cm, class_names, cm_path)

        # 📈 Save per-class F1 bar plot
        save_f1_bar_plot(report_dict, f1_plot_path)

        # 📋 Also show console + inline confusion matrix
        evaluate_model(model, test_loader, device=cfg.device, class_names=class_names)

        if achieved_f1 < 0.70:
            print("⚠️ F1 score below threshold, retrying in 5 seconds...\n")
            time.sleep(5)

        attempts += 1

    if achieved_f1 >= 0.70:
        print(f"\n✅ Success: F1 Score reached {achieved_f1:.4f}")
    else:
        print(f"\n❌ Failed to reach F1 Score ≥ 0.70 after {max_attempts} attempts")

    # 🌐 Generate HTML report
    generate_html_report(
        report_json_path=report_path,
        cm_path=cm_path,
        f1_plot_path=f1_plot_path,
        output_path=html_path
    )

    print(f"\n📄 HTML report generated at: {html_path}")


if __name__ == "__main__":
    main()
