from shroom_classifier.data.utils import image_to_tensor
import torch
import torchvision.transforms as transforms
import pytest

@pytest.mark.parametrize("preprocessor", [None, transforms.ToTensor()])
def test_image_to_tensor(preprocessor) -> None:
    tensor = image_to_tensor("tests/data/placeholder.jpg", preprocesser=preprocessor)
    print(tensor)
    assert isinstance(tensor, torch.Tensor), 'tensor is not a torch.Tensor'

if __name__ == "__main__":
    test_image_to_tensor()
