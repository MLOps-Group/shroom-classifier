import matplotlib.pyplot as plt
import wandb
import numpy as np

def plot_probs(y_true: np.ndarray, probs: np.ndarray) -> wandb.Image:
    ''' Plots the probabilities of the predicted classes.

        Args:
            y_true: True labels (BATCH_SIZE, N_CLASSES)
            probs: Predicted probabilities (BATCH_SIZE, N_CLASSES)

        Returns:
            fig: Figure containing the plot
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.bar(range(len(probs)), probs, label = "Predicted probabilities")
    ax.bar(range(len(probs)), y_true, alpha=0.3, color = "green", width = 5, label = "True label")
    ax.grid(linestyle='--', linewidth=1, axis='y')
    img = wandb.Image(fig)
    plt.close()
    return img