
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# --- Model Loading ---
CHAMPION_MODEL_URI = "models:/pricing-renewal_champion/Production"
CHALLENGER_MODEL_URI = "models:/pricing-renewal_champion/Staging"

def load_model_from_registry(uri):
    try:
        return mlflow.pyfunc.load_model(model_uri=uri)
    except Exception as e:
        logging.warning(f"Could not load model from {uri}: {e}")
        return None

champion_model = load_model_from_registry(CHAMPION_MODEL_URI)
challenger_model = load_model_from_registry(CHALLENGER_MODEL_URI)

app = FastAPI(title="Resilient Pricing API v5")
PREDICTION_LOG_FILE = "prediction_log.jsonl"

class Applicant(BaseModel):
    age: int
    gender: str

@app.post("/predict")
def predict(applicant: Applicant):
    if not champion_model:
        raise HTTPException(status_code=503, detail="Champion model is not available. Service is unhealthy.")
    
    inference_df = pd.DataFrame([applicant.dict()])
    
    champion_prediction = champion_model.predict(inference_df)[0]
    challenger_prediction = None
    if challenger_model:
        try:
            challenger_prediction = challenger_model.predict(inference_df)[0]
        except Exception:
            challenger_prediction = -1

    log_entry = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "request_body": applicant.dict(),
        "champion_prediction": champion_prediction,
        "champion_version": champion_model.metadata.run_id if hasattr(champion_model, 'metadata') else 'unknown',
        "challenger_prediction": challenger_prediction,
        "challenger_version": challenger_model.metadata.run_id if challenger_model and hasattr(challenger_model, 'metadata') else None,
    }
    with open(PREDICTION_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"predicted_premium": champion_prediction}
