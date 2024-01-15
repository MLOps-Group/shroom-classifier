import matplotlib.pyplot as plt
import wandb
from shroom_classifier.visualization.train_plots import plot_probs

def test_plot() -> None:
    y_true = [0.1, 0.2, 0.3, 0.4]
    probs = [0.4, 0.3, 0.2, 0.1]
    img = plot_probs(y_true, probs)
    assert isinstance(img, wandb.Image), 'img is not a wandb.Image'

if __name__ == "__main__":
    test_plot()
