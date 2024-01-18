from fastapi import FastAPI, UploadFile, File 
from fastapi.responses import HTMLResponse
from http import HTTPStatus
from shroom_classifier.predict_model import ShroomPredictor
from shroom_classifier.data.utils import image_to_tensor
import os
import wandb

from omegaconf import OmegaConf
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues, TestColumnDrift, TestAccuracyScore, TestPrecisionScore, TestRecallScore

from monitoring.data_drift import feature_conversion

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


@app.get("/monitoring_test", response_class=HTMLResponse)
async def shroom_monitoring():
    """Request method that returns a monitoring report for testing.
    Testing for data drifting and target drifting.
    """
    
    # Model, Predictor, Data
    config = OmegaConf.load('configs/train_config/train_default_local.yaml')
    model = ShroomClassifierResNet(**config.model)
    predictor = ShroomPredictor("wandb:mlops_papersummarizer/model-registry/shroom_classifier_resnet:latest")
    train_dataset = ShroomDataset(**config.train_dataset, preprocesser=model.preprocesser)
        
    ## Compare N latest sample in original with N 'new' samples 
    df_reference = feature_conversion(model_type="train", N=100)
    df_current_corrupted = feature_conversion(model_type="train_new", N=100)
    
    
    ## Generate testing to get automatic detection 
    data_test = TestSuite(tests=[TestNumberOfMissingValues(), 
                                TestColumnDrift(column_name="avg_brightness", stattest= 't_test', stattest_threshold=0.05 ),
                                TestAccuracyScore(), 
                                TestPrecisionScore(), 
                                TestRecallScore()])
    data_test.run(reference_data=df_reference, current_data=df_current_corrupted)

    data_test.save_html('monitoring_test.html')

    with open("monitoring_test.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/monitoring_exploration", response_class=HTMLResponse)
async def shroom_monitoring():
    """Request method that returns a monitoring report for exploration. 
    """
    
    # Model, Predictor, Data
    config = OmegaConf.load('configs/train_config/train_default_local.yaml')
    model = ShroomClassifierResNet(**config.model)
    predictor = ShroomPredictor("wandb:mlops_papersummarizer/model-registry/shroom_classifier_resnet:latest")
    train_dataset = ShroomDataset(**config.train_dataset, preprocesser=model.preprocesser)
        
    ## Compare N latest sample in original with N 'new' samples 
    df_reference = feature_conversion(model_type="train", N=100)
    df_current_corrupted = feature_conversion(model_type="train_new", N=100)
    
    ## Generate report for exploration and debugging
    report = Report(metrics=[DataDriftPreset(drift_share=0.1), TargetDriftPreset()])
    report.run(reference_data=df_reference, current_data=df_current_corrupted)
    
    report.save_html('monitoring_exploration.html')

    with open("monitoring_exploration.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port = int(os.environ.get("PORT", 8000)))

