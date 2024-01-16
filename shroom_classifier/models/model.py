import timm
from pytorch_lightning import LightningModule
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.loss import BinaryCrossEntropy
from torch import optim
import torch
from shroom_classifier.evaluation.metrics import get_metrics


class ShroomClassifierResNet(LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=num_classes)
        self.preprocesser = create_transform(**resolve_data_config(self.model.pretrained_cfg))
        self.loss = BinaryCrossEntropy()
        self.lr = lr
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # TODO: Change with config
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # TODO: Change with config

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

    def training_step(self, batch, batch_idx):
        images, classes, super_classes = batch
        y_hat = self(images)
        loss = self.loss(y_hat, super_classes)
        if self.logger is not None:
            self.logger.experiment.log({"trainer/global_step": self.global_step})

        # Log metrics
        if self.global_step % 20 == 0:
            if self.logger is not None:
                self.logger.experiment.log({"train/loss": loss})

            probs = torch.nn.functional.softmax(y_hat, dim=1).detach().cpu().numpy()
            prediction = probs.argmax(axis=1)
            targets = super_classes.detach().cpu().numpy()
            acc, precision, recall, f1 = get_metrics(targets.argmax(axis=1), prediction)
            if self.logger is not None:
                self.logger.experiment.log(
                    {"train/accuracy": acc, "train/precision": precision, "train/recall": recall, "train/f1": f1}
                )

        return loss

    def validation_step(self, batch, batch_idx):
        images, classes, super_classes = batch
        y_hat = self(images)
        loss = self.loss(y_hat, super_classes)

        probs = torch.nn.functional.softmax(y_hat, dim=1).detach().cpu().numpy()
        prediction = probs.argmax(axis=1)
        targets = super_classes.detach().cpu().numpy()
        acc, precision, recall, f1 = get_metrics(targets.argmax(axis=1), prediction)

        if self.logger is not None:
            self.log("val/loss", loss)
            self.log("val/accuracy", acc)
            self.log("val/precision", precision)
            self.log("val/recall", recall)
            self.log("val/f1", f1)

        return loss
