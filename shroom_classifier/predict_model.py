import torch
from shroom_classifier import ShroomClassifierResNet
from shroom_classifier.data.utils import image_to_tensor
from shroom_classifier.models import download_model
from PIL import Image
import json
import numpy as np

class ShroomPredictor:
    def __init__(self, model_path, device: torch.device = None):
        # get data info
        info = json.load(open("data/processed/sample.json", "rb"))
        self.super_categories = np.unique([x["supercategory"] for x in info["categories"]])
        
        
        # set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # load model
        self.model_path = model_path
        self.load_model(model_path)
        
        
        
    def load_model(self, model_path):
        # download model if needed
        if model_path.startswith("wandb:"):
            model_path = download_model(model_path[6:])
        
        # load model
        self.model = ShroomClassifierResNet.load_from_checkpoint(model_path, map_location=self.device)
            
    def get_probs(self, image):
        self.model.eval()
        
        if isinstance(image, str):
            image = image_to_tensor(image)
            # image = self.model.preprocesser(image)
            
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
            "probs" : top_k.cpu(), 
            "index": top_k_idx.cpu(),
            "labels" : labels,
        }
    

if __name__ == "__main__":
    # predictor = ShroomPredictor("models/epoch=0-step=2.ckpt")
    predictor = ShroomPredictor("wandb:mlops_papersummarizer/dev/model-p6k4qly6:v2")
    
    probs = predictor.predict("data/processed/sample/10000_Abortiporus_biennis/FVL2009PIC49049490.JPG")
    print(predictor.top_k_preds("data/processed/sample/10000_Abortiporus_biennis/FVL2009PIC49049490.JPG"))
    # print(probs.argmax(axis=1), probs.max(axis=1))
    # print(probs)
    # print("probs sum:", probs.sum())