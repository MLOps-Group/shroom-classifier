from shroom_classifier.predict_model import ShroomPredictor
from shroom_classifier.utils import get_config
import pytest

def test_predict() -> None:
    # get config
    cfg = get_config("test_values.yaml", config_folder="pytest_config")

    predictor = ShroomPredictor(cfg.model.model_path, download_path=cfg.model.download_path)
    probs = predictor.predict(cfg.predict.test_file)
    assert probs is not None, 'No predictions found'

@pytest.mark.parametrize("k", [10, 100])
def test_top_k_preds(k) -> None:
    # get config
    cfg = get_config("test_values.yaml", config_folder="pytest_config")

    predictor = ShroomPredictor(cfg.model.model_path, download_path=cfg.model.download_path)
    top_k_preds = predictor.top_k_preds(cfg.predict.test_file, k=cfg.predict.k)
    assert top_k_preds is not None, 'No top k predictions found'
