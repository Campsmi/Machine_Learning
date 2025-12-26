from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from src.config.settings import FEATURE_COLUMNS
from src.pipelines.inference_pipeline import InferencePipeline


app = FastAPI(title="Song Hit Predictor", version="1.0.0")

pipeline = InferencePipeline("artifacts/models/model.joblib")


class PredictRequest(BaseModel):
    danceability: float = Field(..., ge=0.0, le=1.0)
    energy: float = Field(..., ge=0.0, le=1.0)
    loudness: float
    speechiness: float = Field(..., ge=0.0, le=1.0)
    acousticness: float = Field(..., ge=0.0, le=1.0)
    instrumentalness: float = Field(..., ge=0.0, le=1.0)
    liveness: float = Field(..., ge=0.0, le=1.0)
    valence: float = Field(..., ge=0.0, le=1.0)
    tempo: float
    duration_ms: float
    

class PredictResponse(BaseModel):
    hit_probability: float
    prediction: int  # 1 = hit, 0 = non-hit
    
    

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = pd.DataFrame([req.model_dump()])
    # Ensure column order / completeness
    row = row[FEATURE_COLUMNS]
    out = pipeline.predict(row)
    return out