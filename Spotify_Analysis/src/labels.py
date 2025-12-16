import pandas as pd

def make_is_hit(df, popularity_threshold):
    return (df["popularity"] >= popularity_threshold).astype(int)