from shroom_classifier import ShroomClassifierResNet
from shroom_classifier.data import N_SUPER_CLASSES, ShroomDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
import wandb
import torch
import hydra

# WANDB_ENTITY = "mlops_papersummarizer"      # Wandb entity name
# PROJECT_NAME = "dev"                        # Wandb project name
# MODE = 'online'                             # Wandb mode (online/offline)
# MONITOR = 'accuracy'                        # Validation metric to monitor (saves the best performing model)
# MODEL_DIR = 'models/'                       # Directory to save the model


@hydra.main(config_path="../configs", config_name="config", version_base = None)
def train(cfg):
    # extract train config
    config = cfg.train

    # init model
    model = ShroomClassifierResNet(**config.model)
    
    # set seed
    torch.manual_seed(config.seed)

    # create train dataloader
    train_dataset = ShroomDataset(**config.train_dataset, preprocesser=model.preprocesser)
    train_dataloader = DataLoader(train_dataset, **config.train_dataloader)

    # create val dataloader
    val_dataset = ShroomDataset(**config.val_dataset, preprocesser=model.preprocesser) # Train = Val (for now)
    val_dataloader = DataLoader(val_dataset, **config.val_dataloader)   

    # init callbacks
    checkpoint_callback = ModelCheckpoint(**config.checkpoint_callback)
    lr_monitor = LearningRateMonitor(**config.lr_monitor)

    # init wandb
    wandb_logger = WandbLogger(**config.wandb, config=dict(config))

    trainer = Trainer(**config.trainer, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, train_dataloader, val_dataloader)



if __name__ == "__main__":
    train()