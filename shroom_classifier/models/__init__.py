import wandb
from shroom_classifier.models.model import ShroomClassifierResNet
import torch
import os

os.environ["WANDB_DIR"] = "logs/wandb"
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)


def download_model(full_name: str, download_path: str = "models/"):
    """
    Downloads the model from wandb.

    Parameters
    ----------
    full_name: str
        The full name of the wandb artifact to download.
    download_path: str
        The path to download the artifact to. The model will be downloaded to `download_path/<model_project>/model_<model_id>_<version>/model.ckpt`.

    Returns
    -------
    model_path: str
        The path to the downloaded agent. This is `download_path/model`.
    """

    # reformat model path
    download_path = (
        download_path + full_name.split("/")[1] + "/" + full_name.split("/")[2].replace(":", "_").replace("-", "_")
    )

    api = wandb.Api()  # start wandb api
    artifact = api.artifact(full_name)  # load artifact
    path = artifact.download(download_path)  # download artifact

    return path + "/model.ckpt"


def load_model(model_path, device: torch.device = None, **kwargs):
    """
    Load a model from a checkpoint.
    
    Parameters
    ----------
    model_path: str
        The path to the model checkpoint. If the path starts with "wandb:", the model will be downloaded from wandb.
    device: torch.device
        The device to load the model on.
    **kwargs:
        Additional arguments to pass to the model class.
        
    Returns
    -------
    model: ShroomClassifierResNet
        The loaded model.
    model_path: str
        The path to the loaded model.
    """
    # download model if needed
    if model_path.startswith("wandb:"):
        model_path = download_model(model_path[6:], **kwargs)

    # load model
    model = ShroomClassifierResNet.load_from_checkpoint(model_path, map_location=device)

    return model, model_path


if __name__ == "__main__":
    full_name = "mlops_papersummarizer/dev/model-p6k4qly6:v2"
    download_path = download_model(full_name)
    print(download_path)
