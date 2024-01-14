from shroom_classifier.data.dataset import ShroomDataset
import os
import pytest

@pytest.mark.skipif(not os.path.exists("data/processed/"), reason="Data files not found")
def test_data():
    train_dataset = ShroomDataset("sample", datapath="data/processed/", preprocesser=None)
    val_dataset = ShroomDataset("sample", datapath="data/processed", preprocesser=None)  # Train = Val (for now)

    assert len(train_dataset) != 0
    assert len(val_dataset) != 0
    #assert train_dataset[0][0].shape == (3, 224, 224)
    #assert val_dataset[0][0].shape == (3, 224, 224)


if __name__ == "__main__":
    test_data()
