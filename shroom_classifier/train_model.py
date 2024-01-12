from shroom_classifier import ShroomClassifierResNet
from shroom_classifier.data import N_SUPER_CLASSES, ShroomDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
import wandb


WANDB_ENTITY = "mlops_papersummarizer"      # Wandb entity name
PROJECT_NAME = "dev"                        # Wandb project name
MODE = 'online'                             # Wandb mode (online/offline)
MONITOR = 'accuracy'                        # Validation metric to monitor (saves the best performing model)
MODEL_DIR = 'models/'                       # Directory to save the model



def train():
    model = ShroomClassifierResNet(N_SUPER_CLASSES)
    preprocesser = model.preprocesser

    train_dataset = ShroomDataset("sample", datapath="data/processed/", preprocesser=preprocesser)
    # train_dataset = torch.utils.data.Subset(train_dataset, range(0, 100))
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)

    val_dataset = ShroomDataset("val", datapath="data/raw", preprocesser=preprocesser)
    val_datalaoder = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    wandb_logger = WandbLogger(entity = WANDB_ENTITY, project = PROJECT_NAME, mode = MODE, log_model = "all")

    montior_name = f"val/{MONITOR}"
    checkpoint_callback = ModelCheckpoint(monitor=f"val/{MONITOR}", save_top_k=1, mode='max', dirpath=MODEL_DIR, filename='epoch={epoch}_val_acc={val/accuracy:.2f}', auto_insert_metric_name = False)
    lr_monitor = LearningRateMonitor(logging_interval='step')


    trainer = Trainer(max_epochs=10, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, train_dataloader, val_datalaoder)



if __name__ == "__main__":
    train()