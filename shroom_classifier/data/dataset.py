from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import torch
from os import path
from PIL import Image


_DATA_PATH = "data/processed"

N_CLASSES = 1392
N_SUPER_CLASSES = 418
class ShroomDataset(Dataset):
    ''' Dataset for the mushroom images. 
        The images are saved in the following structure:
            data/
                processed/
                    dataname/
                        class1/
                            image1.jpg
                                ...
                        class2/
                            image2.jpg
                                ...
                        ...
        Args:
            dataname: Name of the dataset. The dataset is saved in data/processed/dataname/
            preprocesser: Preprocesser for the images (if model is specified). 
                        'create_transform(**resolve_data_config(model.pretrained_cfg))' from timm.data library should be used
                        if preprocesser is none reads the images as numpy arrays and returns them as torch tensors
                        (assumes that the images are already preprocessed)
            NB: See example below!
        '''
    def __init__(self, dataname = "sample", preprocesser = None) -> None:
        super().__init__()

        self.info = json.load(open(path.join(_DATA_PATH, "sample.json"), "r"))
        self.images = self.info["images"]
        self.categories = self.info["categories"]
        self.categories_dict = np.load(path.join(_DATA_PATH, "categories.npy"), allow_pickle=True).item()
        self.preprocesser = preprocesser

    def __len__(self) -> int:
        return len(self.info["images"])
    
    def __getitem__(self, index: int):
        ''' Returns image, category and super category for the given index. 

            Args:
                index: Index of the image to be returned.

            Returns:
                img: Image as a torch tensor of shape (3, 224, 224) if preprocesser is None. Otherwise the shape is (3, W, H)
                category: Category of the image as a torch tensor of shape (N_CLASSES)
                super_category: Super category of the image as a torch tensor of shape (N_SUPER_CLASSES)
        '''

        if self.preprocesser is not None:
            img = Image.open(self.images[index]["file_name"])
            img = self.preprocesser(img)

        else:
            img = Image.open(self.images[index]["file_name"])
            img = np.array(img)
            img = torch.tensor(img).permute(2, 0, 1).float()

        category = self.categories[index]["name"]
        super_category = self.categories[index]["supercategory"]

        classes = torch.zeros(N_CLASSES)
        classes[self.categories_dict[category]] = 1

        super_classes = torch.zeros(N_SUPER_CLASSES)
        super_classes[self.categories_dict[super_category]] = 1

        return img, classes, super_classes



if __name__ == '__main__':
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    import timm
    model = timm.create_model('mobilenetv3_large_100', pretrained=True)
    preprocesser = create_transform(**resolve_data_config(model.pretrained_cfg))

    dataset = ShroomDataset("sample", preprocesser)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)
    for img, category, super_category in dataloader:
        print(img.shape)
        print(category.shape)
        print(super_category.shape)
        