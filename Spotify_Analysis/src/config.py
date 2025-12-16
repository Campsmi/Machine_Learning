RANDOM_STATE = 42
TARGET = "is_hit"

DROP_COLUMNS = ["track_id", "track_name", "artist", "album", "release_date", "popularity"]

HIT_THRESHOLD = 51.0

FEATURE_COLUMNS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
    "release_year", "release_month"
]