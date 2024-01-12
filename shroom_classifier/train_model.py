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

WANDB_ENTITY = "mlops_papersummarizer"      # Wandb entity name
PROJECT_NAME = "dev"                        # Wandb project name
MODE = 'online'                             # Wandb mode (online/offline)
MONITOR = 'accuracy'                        # Validation metric to monitor (saves the best performing model)
MODEL_DIR = 'models/'                       # Directory to save the model


@hydra.main(config_path="../configs", config_name="config", version_base = None)
def train(cfg):
    print(cfg.experiments)
    hparams = cfg.experiment.hparams
    wandb_params = cfg.experiment.wandb_params
    model = ShroomClassifierResNet(hparams)


    torch.manual_seed(hparams.seed)
    train_dataset = ShroomDataset(**cfg.train_data, preprocesser=model.preprocesser)
    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=cfg.data.num_workers, seed=hparams.seed)

    val_dataset = ShroomDataset(**cfg.val_data, preprocesser=model.preprocesser) # Train = Val (for now)
    val_datalaoder = DataLoader(val_dataset, batch_size=hparams.batch_size, shuffle=False, num_workers=cfg.data.num_workers)


    wandb_logger = WandbLogger(**wandb_params)

    # montior_name = f"val/{MONITOR}"
    # checkpoint_callback = ModelCheckpoint(monitor=f"val/{MONITOR}", save_top_k=1, mode='max', dirpath=MODEL_DIR, filename='epoch={epoch}_val_acc={val/accuracy:.2f}', auto_insert_metric_name = False)
    # lr_monitor = LearningRateMonitor(logging_interval='step')


    # trainer = Trainer(max_epochs=10, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor])
    # trainer.fit(model, train_dataloader, val_datalaoder)



if __name__ == "__main__":
    train()