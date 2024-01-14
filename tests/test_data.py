from shroom_classifier.data import ShroomDataset

def test_data():
    train_dataset = ShroomDataset("sample", datapath="data/processed/", preprocesser=None)
    val_dataset = ShroomDataset("sample", datapath="data/processed", preprocesser=None) # Train = Val (for now)

    assert len(train_dataset) != 0
    assert len(val_dataset) != 0
    assert train_dataset[0][0].shape == (3, 224, 224)
    assert val_dataset[0][0].shape == (3, 224, 224)


if __name__ == "__main__":
    test_data()