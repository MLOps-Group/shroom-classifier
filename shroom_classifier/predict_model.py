import torch
from shroom_classifier import ShroomClassifierResNet
from shroom_classifier.data.utils import image_to_tensor
from PIL import Image


class ShroomPredictor:
    def __init__(self, model_path, device: torch.device = None):
        # set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # load model
        self.model_path = model_path
        self.load_model(model_path)
        
        
        
    def load_model(self, model_path):
        if model_path.startswith("wandb:"):
            raise NotImplementedError("Wandb loading not implemented")
        else:   
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
        
        print("logits:", logits)
        
        return logits.softmax(dim=1)

    def predict(self, image):
        
        probs = self.get_probs(image)
        probs = probs.detach().cpu().numpy()
        return probs
    

if __name__ == "__main__":
    predictor = ShroomPredictor("models/epoch=0-step=66.ckpt")
    
    probs = predictor.predict("data/processed/sample/10000_Abortiporus_biennis/FVL2009PIC49049490.JPG")
    print(probs)
    print("probs sum:", probs.sum())