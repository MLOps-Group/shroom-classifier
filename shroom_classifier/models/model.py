import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pytorch_lightning import LightningModule
from shroom_classifier.data.dataset import ShroomDataset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.loss import BinaryCrossEntropy 
from torch import optim
import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class ShroomClassifierMobileNetV3Large100(LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=num_classes)
        self.preprocesser = create_transform(**resolve_data_config(self.model.pretrained_cfg))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, classes, super_classes = batch
        y_hat = self(images)
        loss = self.loss(y_hat, super_classes)

        # Log metrics
        if self.global_step % 10 == 0:
            self.logger.experiment.log({"train/loss": loss})
            self.logger.experiment.log({"trainer/step": self.global_step})

            probs = torch.nn.functional.softmax(y_hat, dim=1).detach().cpu().numpy()
            prediction = probs.argmax(axis=1)
            targets    = super_classes.detach().cpu().numpy()
            acc, precision, recall, f1, support = get_metrics(targets.argmax(axis = 1), prediction)
            self.logger.experiment.log({"train/acc": acc,
                                        "train/precision": precision,
                                        "train/recall": recall,
                                        "train/f1": f1,
                                        "train/support": support})

            image = plot_probs(targets[0], probs[0])
            self.logger.experiment.log({"probs": image})

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

def get_metrics(y_true: np.ndarray, y_hat: np.ndarray) -> (float, float, float, float, float):
    ''' Computes classification metrics:
         - Accuracy
        - Precision
        - Recall
        - F1-score
        - Support

        Args:
            y_true: True labels (N_CLASSES)
            y_hat: Predicted labels (N_CLASSES)

        Returns:
            accuracy: Accuracy score
            precision: Precision score
            recall: Recall score
            f1: F1 score
    '''
    
    accuracy = accuracy_score(y_true, y_hat)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_hat, average="macro")

    return accuracy, precision, recall, f1, support


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