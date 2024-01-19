from PIL import Image
import numpy as np
import torch


def image_to_tensor(image_path: str, preprocesser=None) -> torch.Tensor:
    img = Image.open(image_path)

    if preprocesser is not None:
        img = preprocesser(img)
    else:
        img = np.array(img)
        img = torch.tensor(img).permute(2, 0, 1).float()

    return img


def get_labels():
    return np.load("shroom_classifier/data/labels.npy", allow_pickle=True).item()
