import torch
from shroom_classifier.data.utils import image_to_tensor, get_labels
from shroom_classifier.models import load_model
import numpy as np


class ShroomPredictor:
    def __init__(self, model_path: str, device: torch.device = None, **kwargs):
        """
        Create a predictor object for the shroom classifier model.
        
        Parameters
        ----------
        model_path: str
            The path to the model checkpoint. If the path starts with "wandb:", the model will be downloaded from wandb.
        device: torch.device
            A torch device to load the model on.
        **kwargs:
            Additional arguments to pass to the `shroom_classifier.models.load_model()` function.
        """
        
        # get data info
        categories = get_labels()
        self.super_categories = np.array([key for key, val in categories.items() if ' ' not in key])
        
        # set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model
        self.model_path = model_path
        self.load_model(model_path, **kwargs)

    def load_model(self, model_path: str, **kwargs):
        """
        Load a model from a checkpoint.
        
        Parameters
        ----------
        model_path: str
            The path to the model checkpoint. If the path starts with "wandb:", the model will be downloaded from wandb.
        **kwargs:
            Additional arguments to pass to the `shroom_classifier.models.load_model()` function.
        """
        self.model, _ = load_model(model_path, device=self.device, **kwargs)

    def get_probs(self, image):
        self.model.eval()
        if isinstance(image, str):
            image = image_to_tensor(image, preprocesser=self.model.preprocesser)

        # move to device
        image = image.to(self.device)

        # ensure correct shape - add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # predict
        with torch.no_grad():
            logits = self.model(image)
        return logits.softmax(dim=1)

    def predict(self, image):
        probs = self.get_probs(image)
        probs = probs.detach().cpu().numpy()
        return probs

    def top_k_preds(self, image, k=5):
        probs = self.get_probs(image)
        top_k, top_k_idx = torch.topk(probs, k=k, dim=1)
        labels = self.super_categories[top_k_idx.cpu()].squeeze().tolist()
        return {
            "probs": top_k.cpu(),
            "index": top_k_idx.cpu(),
            "labels": labels,
        }


if __name__ == "__main__":
    predictor = ShroomPredictor("wandb:mlops_papersummarizer/model-registry/shroom_classifier_resnet:latest")

    top_k = predictor.top_k_preds("data/processed/sample/10158_Aleuria_aurantia/FVL2009PIC78509508.JPG")
    probs = top_k["probs"]
    index = top_k["index"]
    labels = top_k["labels"]

    print(probs.tolist())
    print(index.tolist())
    print(labels)
    
