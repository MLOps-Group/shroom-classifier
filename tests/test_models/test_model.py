from shroom_classifier.models import load_model
from shroom_classifier.utils import get_config

def test_model():
    # create a config object
    cfg = get_config("test_values.yaml", config_folder="pytest_config")

    # load the model
    model, _ = load_model(cfg.model.model_path, download_path=cfg.model.download_path)

    # check model hyperparameters
    assert model.hparams["num_classes"] == cfg.model.num_classes


if __name__ == "__main__":
    test_model()
