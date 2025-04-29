#init.py
from .config import Config
from .train import train_model
from .evaluate import evaluate_model
from .utils import get_config

def main():
    config = get_config()  # Loads settings from Config class or CLI/JSON/etc.
    model, test_loader, class_names = train_model(config)
    evaluate_model(model, test_loader, config.device, class_names)

if __name__ == "__main__":
    main()


def eval():
    return None
