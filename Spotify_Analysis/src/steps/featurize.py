import pandas as pd
from src.config.settings import HIT_THRESHOLD, TARGET, FEATURE_COLUMNS

def featurize(df):
    df = df.copy()
    
    if "popularity" in df.columns:
        df[TARGET] = (df["popularity"] >= HIT_THRESHOLD).astype(int)
    
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df