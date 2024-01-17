from shroom_classifier.models import ShroomClassifierResNet
from shroom_classifier.data import ShroomDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import hydra
import os


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg):
    # extract train config
    config = cfg.train_config

    # init model
    model = ShroomClassifierResNet(**config.model)

    # set seed
    torch.manual_seed(config.seed)

    # create train dataloader
    train_dataset = ShroomDataset(**config.train_dataset, preprocesser=model.preprocesser)
    train_dataloader = DataLoader(train_dataset, **config.train_dataloader)

    # create val dataloader
    val_dataset = ShroomDataset(**config.val_dataset, preprocesser=model.preprocesser)  # Train = Val (for now)
    val_dataloader = DataLoader(val_dataset, **config.val_dataloader)

    # init callbacks
    checkpoint_callback = ModelCheckpoint(**config.checkpoint_callback)
    lr_monitor = LearningRateMonitor(**config.lr_monitor)

    # init wandb
    os.makedirs(config.wandb.save_dir, exist_ok=True)
    wandb_logger = WandbLogger(**config.wandb, config=eval(str(config)))

    # init trainer
    trainer = Trainer(**config.trainer, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
