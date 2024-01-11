from shroom_classifier import ShroomClassifierMobileNetV3Large100
from shroom_classifier.data import N_SUPER_CLASSES, ShroomDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger



def train():
    model = ShroomClassifierMobileNetV3Large100(N_SUPER_CLASSES)
    preprocesser = model.preprocesser

    train_dataset = ShroomDataset("train", datapath="data/raw", preprocesser=preprocesser)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    val_dataset = ShroomDataset("val", datapath="data/raw", preprocesser=preprocesser)
    val_datalaoder = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=8)

    model.train()
    wandb_logger = WandbLogger()
    trainer = Trainer(max_epochs=10, logger=wandb_logger)
    trainer.fit(model, train_dataloader)



if __name__ == "__main__":
    train()