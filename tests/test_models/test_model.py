from shroom_classifier.models import load_model
import hydra

def test_model():
    # create a config object
    with hydra.initialize(version_base=None, config_path="../../configs/pytest_config"):
        cfg = hydra.compose(config_name="test_values.yaml")#, overrides=["app.user=test_user"])

    # load the model
    model, _ = load_model("wandb:mlops_papersummarizer/dev/model-dct9b3c3:v3")

    # check model hyperparameters
    assert model.hparams["num_classes"] == cfg.model.num_classes


# from shroom_classifier.models.model import ShroomClassifierResNet


# def test_model() -> None:
#     model = ShroomClassifierResNet(num_classes=10)
#     assert model is not None

if __name__ == "__main__":
    test_model()
