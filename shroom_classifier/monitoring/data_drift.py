from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues, TestColumnDrift, TestAccuracyScore, TestPrecisionScore, TestRecallScore
from shroom_classifier.data import ShroomDataset
from shroom_classifier import ShroomClassifierResNet
from omegaconf import OmegaConf
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
from shroom_classifier.predict_model import ShroomPredictor

## Sammenligner N sidste samples i original med N sidste samples i nye
## Sammenlign på average brightness
config = OmegaConf.load('configs/train_config/train_default_local.yaml')

# Model, Predictor, Data
model = ShroomClassifierResNet(**config.model)
predictor = ShroomPredictor("wandb:mlops_papersummarizer/model-registry/shroom_classifier_resnet:latest")
train_dataset = ShroomDataset(**config.train_dataset, preprocesser=model.preprocesser)

def feature_conversion(model_type: str, N: int):
    """ 
    feature_conversion returns new dataframe with features

    Args:
        model_type (str): either 'train' or 'train_new' for latest simulated data  
        N: number of data points to test
        
    Return:
        New dataframe with features 
    """
    
    # Initialize lists to store features for each image
    average_brightness_list = []
    predictions_list = [] 
    targets_list = []
    
        
    for i in range(N):
        if model_type == "train":
            i = -1*i
        # Get the i-th sample
        sample = train_dataset[i][0]

        # Convert PyTorch tensor to NumPy array
        image_np = sample.numpy()

        # Reshape the image array to (height, width, channels) for compatibility with PIL
        image_np = np.transpose(image_np, (1, 2, 0))

        # Convert NumPy array to PIL Image
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        
        if model_type == "train_new":
            brightness_factor = 5.0 
            enhancer = ImageEnhance.Brightness(image_pil)
            brightened_image = enhancer.enhance(brightness_factor)

            # Convert the image back to NumPy array
            brightened_image_np = np.array(brightened_image) / 255.0

            # Convert the brightened image to grayscale for simplicity
            image_gray = Image.fromarray((brightened_image_np * 255).astype(np.uint8)).convert('L')
        else:
            # Convert the image to grayscale for simplicity
            image_gray = image_pil.convert('L')

        # Calculate average brightness
        average_brightness = np.mean(image_gray)

        # Append to list
        average_brightness_list.append(average_brightness)
        
        predict = predictor.predict(train_dataset[i][0]).argmax()
        predictions_list.append(predict)
        
        targets = train_dataset[i][2]
        targets = targets.detach().cpu().numpy().argmax()
        targets_list.append(targets)

    # Convert to tabular data, requirement by Evidently
    data = {"avg_brightness": np.array(average_brightness_list), "target":np.array(targets_list), "prediction": np.array(predictions_list)} 
    df = pd.DataFrame(data) 
    
    if model_type == "new_train":
        df.loc[1:2,"avg_brightness"]=np.nan

    return df


## Target drift: Check if model is overpredicting/underpredicting certain classes / dist of pred values differs 
#from ground truth. 
# + data drifting

df_reference = feature_conversion(model_type="train", N=100)
df_current_corrupted = feature_conversion(model_type="train_new", N=100)

## Generate report for exploration and debugging
report = Report(metrics=[DataDriftPreset(drift_share=0.1), TargetDriftPreset()])
report.run(reference_data=df_reference, current_data=df_current_corrupted)
report.save_html('report_exploration.html')


## Generate testing to get automatic detection 
data_test = TestSuite(tests=[TestNumberOfMissingValues(), 
                             TestColumnDrift(column_name="avg_brightness", stattest= 't_test', stattest_threshold=0.05 ),
                             TestAccuracyScore(), 
                             TestPrecisionScore(), 
                             TestRecallScore()])
data_test.run(reference_data=df_reference, current_data=df_current_corrupted)
#print(data_test.as_dict())
data_test.save_html('report_test.html')

