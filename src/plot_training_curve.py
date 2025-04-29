import json
import matplotlib.pyplot as plt

def plot_training_curve(log_path, output_path='training_curve.png'):
    with open(log_path, 'r') as f:
        logs = json.load(f)

    epochs = [entry['epoch'] for entry in logs]
    losses = [entry['loss'] for entry in logs]
    lrs = [entry['lr'][0] if isinstance(entry['lr'], list) else entry['lr'] for entry in logs]

    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color)
    ax2.plot(epochs, lrs, color=color, linestyle='--', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Training Loss and Learning Rate Over Epochs")
    plt.savefig(output_path)
    plt.close()

    
if __name__ == "__main__":
    print("Generating plot...")
    plot_training_curve("training_log.json")
    print("Plot saved!")

