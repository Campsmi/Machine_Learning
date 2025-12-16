import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.settings import TARGET, TEST_SIZE, RANDOM_STATE, FEATURE_COLUMNS

def split(df):
    
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()
    
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)