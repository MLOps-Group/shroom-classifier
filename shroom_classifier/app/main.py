from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from shroom_classifier.predict_model import ShroomPredictor
from shroom_classifier.data.utils import image_to_tensor
import os
import wandb

app = FastAPI()


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.get("/envvars/")
def read_envvars():
    """Get an item by id."""
    return {"envvars": dict(os.environ)}

@app.get("/wandb_model/")
async def read_model():
    """Get a model by id."""
    print("Downloading model...")
    full_name = "mlops_papersummarizer/model-registry/shroom_classifier_resnet:latest"
    download_path = "models/"
    
    download_path = (
        download_path + full_name.split("/")[1] + "/" + full_name.split("/")[2].replace(":", "_").replace("-", "_")
    )

    api = wandb.Api()  # start wandb api
    artifact = api.artifact(full_name)  # load artifact
    path = artifact.download(download_path)  # download artifact

    return {"wandb_model_id": full_name, "path": path + "/model.ckpt"}



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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port = int(os.environ.get("PORT", 8000)))