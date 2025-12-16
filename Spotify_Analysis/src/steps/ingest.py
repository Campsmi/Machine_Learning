import pandas as pd
from src.config.settings import DATA_PATH

def ingest():
    return pd.read_csv(DATA_PATH)