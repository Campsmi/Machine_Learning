import pandas as  pd
import joblib

from src.config.settings import FEATURE_COLUMNS
from src.steps.featurize import featurize


class InferencePipeline:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        
    
    def predict(self, input_df):
        
        df = input_df.copy()
        
        missing = set(FEATURE_COLUMNS) - set(df.columns)
        
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        df = featurize(df)
        
        X = df[FEATURE_COLUMNS]
        proba = self.model.predict_proba(X)[:,1]
        
        return {
            "hit_probability": float(proba[0]),
            "prediction": int(proba[0] >= 0.5)
        }