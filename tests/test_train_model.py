# from shroom_classifier.models.model import ShroomClassifierResNet
# from shroom_classifier.data.dataset import ShroomDataset
# from torch.utils.data import DataLoader
# from pytorch_lightning import Trainer
# import hydra
# import os
# import pytest

# #@pytest.mark.skipif(not os.path.exists("../configs"), reason="Config files not found")
# def test_train() -> None:
#     with hydra.initialize(version_base= None, config_path="../configs/train_config/"):
#         config = hydra.compose(config_name="train_default")
#     # extract train config

#     # init model
#     model = ShroomClassifierResNet(**config.model)

#     train_dataset = ShroomDataset(**config.train_dataset, preprocesser=model.preprocesser)
#     train_dataloader = DataLoader(train_dataset, **config.train_dataloader)

#     # create val dataloader
#     val_dataset = ShroomDataset(**config.val_dataset, preprocesser=model.preprocesser)  # Train = Val (for now)
#     val_dataloader = DataLoader(val_dataset, **config.val_dataloader)


#     trainer = Trainer(**config.trainer, logger=False, callbacks=None)
#     #trainer.fit(model, train_dataloader, val_dataloader) #TODO: Make it so it fits and trains on 1 image
#     assert trainer is not None

from shroom_classifier.train_model import train
from shroom_classifier.utils import get_config

def test_train() -> None:
    class TrainConfig:
        train_config = get_config("train_model_test", "pytest_config")
        
    train(TrainConfig)

    
