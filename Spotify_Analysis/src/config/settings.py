from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "spotify_analysis_dataset.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2


HIT_THRESHOLD = 51.0
TARGET = "is_hit"

DROP_COLUMNS = [
    "track_id",
    "track_name",
    "artist",
    "album",
    "release_date",
    "popularity",
]

FEATURE_COLUMNS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
]