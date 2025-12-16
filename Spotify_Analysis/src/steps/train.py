from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.config.settings import RANDOM_STATE


def build_models():
    
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])
    
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE
        ))
    ])
    
    return {
        "log_reg": lr,
        "random_forest": rf
    }