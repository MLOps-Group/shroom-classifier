from shroom_classifier.predict_model import ShroomPredictor

def test_predict(): 
    predictor = ShroomPredictor("wandb:mlops_papersummarizer/dev/model-dct9b3c3:v3")
    probs = predictor.predict("data/processed/sample/10000_Abortiporus_biennis/FVL2009PIC49049490.JPG")
    assert probs is not None

def test_top_k_preds():
    predictor = ShroomPredictor("wandb:mlops_papersummarizer/dev/model-dct9b3c3:v3")
    top_k_preds = predictor.top_k_preds("data/processed/sample/10000_Abortiporus_biennis/FVL2009PIC49049490.JPG")
    assert top_k_preds is not None


