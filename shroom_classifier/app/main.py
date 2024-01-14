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
    predictor = ShroomPredictor("models/dev/model_dct9b3c3_v3/model.ckpt")

    # Read the image file
    os.makedirs(".tmp/images", exist_ok=True)
    with open(".tmp/images/image.jpg", "wb") as image:
        image_data = await file.read()
        image.write(image_data)
        image.close()

    # Convert the image to a tensor
    image = image_to_tensor(".tmp/images/image.jpg")

    # Make a prediction
    top_k_preds = predictor.top_k_preds(image, k=k)
    top_k_preds["probs"] = top_k_preds["probs"].tolist()
    top_k_preds["index"] = top_k_preds["index"].tolist()

    # Delete the image file
    os.remove(".tmp/images/image.jpg")

    # Return the prediction
    return {"filename": file.filename, "top_k_preds": top_k_preds}
