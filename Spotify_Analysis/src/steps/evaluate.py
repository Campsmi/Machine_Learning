from __future__ import annotations
import json
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, classification_report

from src.config.settings import RANDOM_STATE

SCORING = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}


@dataclass
class EvalSummary:
    model:str
    cv_mean: dict
    cv_std: dict
    test_roc_auc: dict
    test_report: dict
    

def evaluate_model(model, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING)

    cv_mean = {k: float(np.mean(scores[f"test_{k}"])) for k in SCORING.keys()}
    cv_std = {k: float(np.std(scores[f"test_{k}"])) for k in SCORING.keys()}

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    test_auc = float(roc_auc_score(y_test, proba))
    report = classification_report(y_test, pred, output_dict=True)

    return {
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "test_roc_auc": test_auc,
        "test_report": report,
    }