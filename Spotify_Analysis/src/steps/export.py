import json
from dataclasses import asdict, dataclass
import joblib

from src.config.settings import MODELS_DIR, REPORTS_DIR, FEATURE_COLUMNS, HIT_THRESHOLD, RANDOM_STATE

@dataclass
class Metadata:
    model_name: str
    feature_columns: list[str]
    hit_threshold: float
    random_state: int

def export_artifacts(best_name: str, best_model, eval_results: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODELS_DIR / "model.joblib")

    meta = Metadata(
        model_name=best_name,
        feature_columns=FEATURE_COLUMNS,
        hit_threshold=HIT_THRESHOLD,
        random_state=RANDOM_STATE
    )
    (MODELS_DIR / "metadata.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(eval_results, indent=2), encoding="utf-8")