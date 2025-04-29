# src/run.py
import argparse
from Pipeline.config import Config
from Pipeline.train import train_model
from Pipeline.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate dementia classifier")
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()

    cfg = Config()
    
    # Train and get model + test data
    model, test_loader, class_names = train_model(resume=args.resume)

    # Evaluate the trained model
    evaluate_model(model, test_loader, device=cfg.device, class_names=class_names)

if __name__ == "__main__":
    main()
