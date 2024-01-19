# Shroom Classifier

Mushroom foraging has gained popularity as a recreational activity, yet its seemingly harmless nature hides potential dangers, as some fungi can be fatally poisonous. 
Understanding the risks associated with collecting and consuming mushrooms is crucial. To enhance the safety of mushroom hunting enthusiasts, we present ShroomClassifier, an innovative image classification framework built on the powerful TIMM (Transfer Learning with Image Models) library from Hugging Face (https://huggingface.co/timm).

## Features
Powered by TIMM: Leveraging the cutting-edge capabilities of the TIMM library, ShroomClassifier ensures state-of-the-art performance in image classification. TIMM's pre-trained models enable accurate and efficient identification of various fungi species.

Accessible Data: ShroomClassifieris finetuned on a comprehensive dataset of labeled images of fungi, ensuring a diverse and reliable training set. The dataset can be easily accessed and downloaded from: https://github.com/visipedia/fgvcx_fungi_comp#data

## Try it out

To try out the Shroom Classifier go to https://shroom-classifier.streamlit.app/ or scan the QR code below:

![QR code](shroom_classifier/app/app_qr_code.png)


## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```


