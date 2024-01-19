import wandb
import os
from fastapi import FastAPI, UploadFile, File
# from shroom_classifier.predict_model import ShroomPredictor
# from shroom_classifier.data.utils import image_to_tensor

os.environ["WANDB_DIR"] = "logs/wandb"
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

# check if wandb is logged in

# if not wandb.api.api_key:
#     wandb.login(key = os.environ["WANDB_API_KEY"])


app = FastAPI()


@app.get("/")
def read_root():
    """Root endpoint."""
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    """Get an item by id."""
    return {"item_id": item_id}


@app.get("/envvars/")
def read_envvars():
    """Get an item by id."""
    return {"envvars": dict(os.environ)}



@app.get("/cloudrun/")
def read_cloudrun():
    """Get an item by id."""
    # return {"cloudrun": ON_CLOUD_RUN}
    return {"cloudrun": os.environ.get("GOOGLE_CLOUD_PROJECT", False)}

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



@app.post("/predict/")
async def predict(file: UploadFile = File(...), k: int = 5):
    # Load the model
    # predictor = ShroomPredictor("wandb:mlops_papersummarizer/model-registry/shroom_classifier_resnet:latest")

    # Read the image file
    os.makedirs(".tmp/images", exist_ok=True)
    with open(".tmp/images/image.jpg", "wb") as image:
        image_data = await file.read()
        image.write(image_data)
        image.close()

    # Convert the image to a tensor
    # image = image_to_tensor(".tmp/images/image.jpg")

    # Make a prediction
    # top_k_preds = predictor.top_k_preds(image, k=k)
    # top_k_preds["probs"] = top_k_preds["probs"].tolist()
    # top_k_preds["index"] = top_k_preds["index"].tolist()
    top_k_preds = {"probs": [0.1, 0.2, 0.3, 0.4, 0.5], "index": [0, 1, 2, 3, 4]}

    # Delete the image file
    # os.remove(".tmp/images/image.jpg")

    # Return the prediction
    return {"filename": file.filename, "top_k_preds": top_k_preds}



if __name__ == "__main__":
    # if ON_CLOUD_RUN:
    #     import gunicorn
    #     gunicorn.run(app, host="
    # else:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port = int(os.environ.get("PORT", 8080)))
