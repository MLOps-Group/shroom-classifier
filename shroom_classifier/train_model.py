from shroom_classifier import ShroomClassifierResNet
from shroom_classifier.data import N_SUPER_CLASSES, ShroomDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


WANDB_ENTITY = "mlops_papersummarizer"
PROJECT_NAME = "dev"



def train():
    model = ShroomClassifierResNet(N_SUPER_CLASSES)
    preprocesser = model.preprocesser

    train_dataset = ShroomDataset("sample", datapath="data/processed/", preprocesser=preprocesser)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)

    # val_dataset = ShroomDataset("val", datapath="data/processed", preprocesser=preprocesser)
    # val_datalaoder = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=8)

    model.train()
    wandb_logger = WandbLogger(entity = WANDB_ENTITY, project = PROJECT_NAME)
    trainer = Trainer(max_epochs=10, logger=wandb_logger)
    # trainer.fit(model, train_dataloader, val_datalaoder)
    trainer.fit(model, train_dataloader)



if __name__ == "__main__":
    train()