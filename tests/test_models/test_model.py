from shroom_classifier.models import load_model
import hydra

@hydra.main(config_path="../../configs/", config_name="test_values.yaml")
def test_model(cfg):
    model, _ = load_model("wandb:mlops_papersummarizer/dev/model-dct9b3c3:v3")
    assert model.hparams["num_classes"] == cfg.model.num_classes
    
if __name__ == "__main__":
    test_model()