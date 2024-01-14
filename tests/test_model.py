from shroom_classifier.models.model import ShroomClassifierResNet


def test_model() -> None:
    model = ShroomClassifierResNet(num_classes=10)
    assert model is not None
