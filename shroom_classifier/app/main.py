from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from shroom_classifier.predict_model import ShroomPredictor
from shroom_classifier.data.utils import image_to_tensor
import os

app = FastAPI()


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict")
async def predict(file: UploadFile = File(...), k: int = 5):
    # Load the model
    
    predictor = ShroomPredictor("wandb:mlops_papersummarizer/model-registry/shroom_classifier_resnet:latest")
    # Read the image file
    os.makedirs(".tmp/images", exist_ok=True)
    with open(".tmp/images/image.jpg", "wb") as image:
        image_data = await file.read()
        image.write(image_data)
        image.close()

    # Convert the image to a tensor
    img = image_to_tensor(".tmp/images/image.jpg", preprocesser=predictor.model.preprocesser)

    # Make a prediction
    top_k_preds = predictor.top_k_preds(img, k=k)
    top_k_preds["probs"] = top_k_preds["probs"].tolist()
    top_k_preds["index"] = top_k_preds["index"].tolist()
    # Delete the image file
    os.remove(".tmp/images/image.jpg")

    # Return the prediction
    return {"filename": file.filename, "top_k_preds": top_k_preds}
