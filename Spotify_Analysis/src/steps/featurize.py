import pandas as pd
from src.config.settings import HIT_THRESHOLD, TARGET, FEATURE_COLUMNS

def featurize(df):
    df = df.copy()
    
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    
    df[TARGET] = (df["popularity"] >= HIT_THRESHOLD).astype(int)
    
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df