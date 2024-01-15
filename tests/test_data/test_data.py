from shroom_classifier.data.dataset import ShroomDataset
from shroom_classifier.data.make_dataset import unpack_data, categories_dictionary
import os
import pytest

@pytest.mark.skipif(not os.path.exists("data/processed/"), reason="Data files not found")
def test_data():
    train_dataset = ShroomDataset("sample", datapath="data/processed/", preprocesser=None)
    val_dataset = ShroomDataset("sample", datapath="data/processed", preprocesser=None)  # Train = Val (for now)

    assert len(train_dataset) != 0
    assert len(val_dataset) != 0
    #assert train_dataset[0][0].shape == (3, 224, 224) #TODO: Fails in Github Actions due to filename 
    #assert val_dataset[0][0].shape == (3, 224, 224) #TODO: Fails in Github Actions due to filename 

#def test_unpack_data() -> None:
 #   unpack_data("sample.tar.gz") #TODO: make a small tar file fot testing
  #  assert os.path.exists("data/raw/sample/10000_Abortiporus_biennis/FVL2009PIC49049490.JPG")
   # assert os.path.exists("data/raw/sample/10000_Abortiporus_biennis/FVL2009PIC49049490.json")
    #del os.remove("data/raw/sample/10000_Abortiporus_biennis/FVL2009PIC49049490.JPG")
    #del os.remove("data/raw/sample/10000_Abortiporus_biennis/FVL2009PIC49049490.json")

def test_categories_dictionary() -> None:
    dictionary = categories_dictionary("data/")
    print(dictionary)
    assert dictionary == {'id': 10000, 'name': 'Placeholder', 'supercategory': 'fungi'}


if __name__ == "__main__":
    test_data()
    test_categories_dictionary()
