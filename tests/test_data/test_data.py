from shroom_classifier.data.dataset import ShroomDataset
import os
import pytest
from shroom_classifier.utils import get_config

@pytest.mark.skipif(not os.path.exists("data/processed/"), reason="Data files not found")
def test_data():
    # get config
    cfg = get_config("test_values.yaml", config_folder="pytest_config")
    
    # create datasets
    train_dataset = ShroomDataset(dataname = cfg.data.train.dataname, datapath=cfg.data.train.datapath, preprocesser=None)
    val_dataset = ShroomDataset(dataname = cfg.data.val.dataname, datapath=cfg.data.val.datapath, preprocesser=None)  # Train = Val (for now)

    # check that datasets are not empty
    assert len(train_dataset) != 0, "Train dataset is empty"
    assert len(val_dataset) != 0, "Val dataset is empty"
    
    # check that datasets have the correct shape
    assert train_dataset[0][0].shape == (3, 224, 224), f"Train dataset has shape {train_dataset[0][0].shape} instead of (3, 224, 224)"
    assert val_dataset[0][0].shape == (3, 224, 224), f"Val dataset has shape {val_dataset[0][0].shape} instead of (3, 224, 224)"


if __name__ == "__main__":
    test_data()
