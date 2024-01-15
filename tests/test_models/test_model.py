import unittest
import torch
from shroom_classifier.models.model import ShroomClassifierResNet
from shroom_classifier.models import load_model
from shroom_classifier.utils import get_config

def test_model() -> None:
    # create a config object
    cfg = get_config("test_values.yaml", config_folder="pytest_config")

    # load the model
    model, _ = load_model(cfg.model.model_path, download_path=cfg.model.download_path)

    # check model hyperparameters
    assert model.hparams["num_classes"] == cfg.model.num_classes

class TestShroomClassifierResNet(unittest.TestCase):
    def setUp(self) -> None:
        # Initialize the model with a dummy number of classes
        self.model = ShroomClassifierResNet(num_classes=2)

    def test_forward_pass(self) -> None:
        # Test the forward pass of the model
        dummy_input = torch.randn((1, 3, 224, 224))  # Assuming input shape (batch_size, channels, height, width)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (1, 2))  # Adjust the shape based on your actual number of classes

    def test_configure_optimizers(self) -> None:
        # Test the configure_optimizers method
        optimizer, lr_scheduler, monitor = self.model.configure_optimizers()

        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(lr_scheduler)

    def test_training_step(self) -> None:
        # Test the training_step method
        dummy_batch = (torch.randn((32, 3, 224, 224)), torch.randint(2, (32,)), torch.nn.functional.one_hot(torch.randint(2, (32,)), num_classes=2).float())
        loss = self.model.training_step(dummy_batch, 0)

        self.assertIsNotNone(loss)
        

    def test_validation_step(self) -> None:
        # Test the validation_step method
        dummy_batch = (torch.randn((32, 3, 224, 224)), torch.randint(2, (32,)), torch.nn.functional.one_hot(torch.randint(2, (32,)), num_classes=2).float())
        loss = self.model.validation_step(dummy_batch, 0)

        self.assertIsNotNone(loss)


if __name__ == "__main__":
    test_model()
    unittest.main()
