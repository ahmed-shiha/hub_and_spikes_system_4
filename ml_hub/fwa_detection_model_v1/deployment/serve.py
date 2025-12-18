
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import logging
import json
from datetime import datetime
import os
import sys
from pathlib import Path

# Add project root for robust imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.online_features import get_member_claims_in_last_hour

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Loading ---
MODEL_NAME = "fwa-detection-model"
CHAMPION_URI = f"models:/{MODEL_NAME}/Production"
CHALLENGER_URI = f"models:/{MODEL_NAME}/Staging"
PREDICTION_LOG_FILE = "fwa_prediction_log.jsonl"

def load_model_from_registry(uri, model_name):
    try:
        # MLflow's pyfunc loader automatically handles signature enforcement
        return mlflow.pyfunc.load_model(model_uri=uri)
    except Exception as e:
        logging.warning(f"Could not load {model_name} from {uri}: {e}")
        return None

champion_model = load_model_from_registry(CHAMPION_URI, "Champion")
challenger_model = load_model_from_registry(CHALLENGER_URI, "Challenger")

app = FastAPI(title="FWA Detection API v1 (Resilient)")

class Claim(BaseModel):
    claim_id: str
    member_id: str
    avg_cost_per_claim: float
    weekend_billing_ratio: float
    high_cost_claim_ratio: float

@app.post("/predict_fraud")
def predict_fraud(claim: Claim):
    if not champion_model:
        raise HTTPException(status_code=503, detail="Champion model not loaded. Service is unhealthy.")

    # 1. Fetch online feature
    claims_last_hour = get_member_claims_in_last_hour(claim.member_id)

    # 2. Combine with offline/request features
    features_dict = {
        'avg_cost_per_claim': claim.avg_cost_per_claim,
        'weekend_billing_ratio': claim.weekend_billing_ratio,
        'high_cost_claim_ratio': claim.high_cost_claim_ratio,
        # The online feature is now part of the feature vector
        'claims_in_last_hour': claims_last_hour
    }
    inference_df = pd.DataFrame([features_dict])

    # 3. Predict with Champion
    try:
        # For classifiers, pyfunc.predict() often returns the class label. 
        # To get probabilities, the model needs to be saved with a signature that specifies this.
        # Here we simulate that the output is the probability of the positive class (1).
        champion_proba = champion_model.predict(inference_df)[0]
    except Exception as e:
        logging.error(f"Champion prediction failed: {e}")
        # In a real system, you might want a fallback to a default score or business rule.
        raise HTTPException(status_code=500, detail="Champion prediction failed.")

    # 4. Predict with Challenger (if it exists) for logging
    challenger_proba = None
    if challenger_model:
        try:
            challenger_proba = challenger_model.predict(inference_df)[0]
        except Exception:
            challenger_proba = -1.0 # Sentinel value for error

    # 5. Log for Observability
    log_entry = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "request_body": claim.dict(),
        "champion_prediction_proba": champion_proba,
        "champion_version": champion_model.metadata.run_id if hasattr(champion_model, 'metadata') else 'unknown',
        "challenger_prediction_proba": challenger_proba,
        "challenger_version": challenger_model.metadata.run_id if challenger_model and hasattr(challenger_model, 'metadata') else None,
    }
    # In a real system, this would log to a scalable service like Kafka, S3, or a dedicated logging database.
    with open(PREDICTION_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "claim_id": claim.claim_id,
        "fraud_probability": champion_proba,
        "model_version": champion_model.metadata.run_id if hasattr(champion_model, 'metadata') else 'unknown'
    }

@app.get("/health")
def health_check():
    if champion_model:
        return {"status": "ok", "champion_model_version": champion_model.metadata.run_id if hasattr(champion_model, 'metadata') else 'unknown'}
    else:
        raise HTTPException(status_code=503, detail="Champion model is not loaded.")

