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


        if self.global_step % 10 == 0:
            self.logger.experiment.log({"train_loss": loss})
            self.logger.experiment.log({"trainer/step": self.global_step})
            logits = y_hat[0]
            probs = torch.exp(logits) / torch.exp(logits).sum()
            probs = probs.detach().cpu().numpy()    
            prediction = torch.argmax(y_hat.sigmoid(), axis=0).detach().cpu().numpy()
            targets    = torch.argmax(super_classes, axis=0).detach().cpu().numpy()

            acc = np.mean(prediction == targets)
            self.logger.experiment.log({"train_acc": acc})
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.bar(range(len(probs)), probs * 3)
            y_true = super_classes[0].detach().cpu().numpy()
            ax.bar(range(len(probs)), y_true, alpha=0.3, color = "green", width = 5)

            
            image = wandb.Image(fig)
            self.logger.experiment.log({"probs": image})
            plt.close()


        return loss
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    


    