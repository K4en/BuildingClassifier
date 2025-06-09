import torch
import matplotlib.pyplot as plt
import os

def save_model(model, path):
    """Saves the model to the given path"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def plot_training_loss(loss_values, save_path=None):
    """Plots training loss over epochs"""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_values, marker='o', label='Training loss')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Loss curves saved to {save_path}")
    plt.close()

def count_parameters(model):
    """Counts the number of parameters in the given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)