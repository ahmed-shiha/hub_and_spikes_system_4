
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add project root for robust imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Config and Model Loading ---
config = load_config()
MODEL_NAME = "stp-automation-model"
PREDICTION_LOG_FILE = "stp_prediction_log.jsonl"

def load_model_and_artifacts(stage):
    """Loads a model and its class mapping artifact from a specific stage in MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        version_info = client.get_latest_versions(MODEL_NAME, stages=[stage])[0]
        model_uri = f"models:/{MODEL_NAME}/{version_info.version}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        
        # Download the class mapping artifact associated with this model run
        artifact_path = client.download_artifacts(version_info.run_id, "class_mapping.json")
        with open(artifact_path, 'r') as f:
            class_mapping = {int(k): v for k, v in json.load(f).items()}
            
        logging.info(f"Successfully loaded model version '{version_info.version}' from stage '{stage}'.")
        return model, class_mapping, version_info
    except Exception as e:
        logging.warning(f"Could not load model from stage '{stage}': {e}")
        return None, None, None

champion_model, champion_mapping, champion_version = load_model_and_artifacts("Production")
challenger_model, challenger_mapping, challenger_version = load_model_and_artifacts("Staging")


app = FastAPI(title="STP Triage API v1 (Resilient)")

class Claim(BaseModel):
    claim_id: str
    submitted_amount: float
    has_pre_auth: int
    service_is_covered: int
    is_complex_diag: int

@app.post("/triage_claim")
def triage_claim(claim: Claim):
    if not all([champion_model, config, champion_mapping]):
        raise HTTPException(status_code=503, detail="Champion model/config not loaded. Service is unavailable.")

    # In a real system, features would be fetched from online/offline stores
    features_dict = {
        'net_amount': claim.submitted_amount, 
        'has_pre_auth': claim.has_pre_auth,
        'service_is_covered': claim.service_is_covered,
        'is_complex_diag': claim.is_complex_diag
    }
    inference_df = pd.DataFrame([features_dict])
    
    # --- Champion Prediction ---
    try:
        champ_proba = champion_model.predict(inference_df)
        champ_idx = np.argmax(champ_proba, axis=1)[0]
        champ_label = champion_mapping.get(champ_idx, "UNKNOWN")
        champ_conf = champ_proba[0, champ_idx]
    except Exception as e:
        logging.error(f"Champion prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Champion prediction failed.")

    # --- Challenger Prediction (for logging/analysis) ---
    challenger_label, challenger_conf = None, None
    if challenger_model and challenger_mapping:
        try:
            chall_proba = challenger_model.predict(inference_df)
            chall_idx = np.argmax(chall_proba, axis=1)[0]
            challenger_label = challenger_mapping.get(chall_idx, "UNKNOWN")
            challenger_conf = chall_proba[0, chall_idx]
        except Exception:
            challenger_label, challenger_conf = "ERROR", -1.0

    # --- Apply Business Logic to Champion's Prediction ---
    thresholds = config['business_thresholds']
    final_decision = "PEND_FOR_REVIEW"
    if champ_label == 'APPROVE' and champ_conf >= thresholds['auto_approve_confidence']:
        final_decision = 'AUTO_APPROVED'
    elif champ_label == 'DENY' and champ_conf >= thresholds['auto_deny_confidence']:
        final_decision = 'AUTO_DENIED'
        
    # --- Log for Observability ---
    log_entry = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "claim_id": claim.claim_id,
        "champion_prediction": champ_label,
        "champion_confidence": float(champ_conf),
        "champion_version": champion_version.version if champion_version else 'unknown',
        "challenger_prediction": challenger_label,
        "challenger_confidence": float(challenger_conf) if challenger_conf is not None else None,
        "challenger_version": challenger_version.version if challenger_version else None,
        "final_decision": final_decision
    }
    with open(PREDICTION_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "claim_id": claim.claim_id,
        "final_decision": final_decision,
        "model_version": champion_version.version if champion_version else 'unknown'
    }

@app.get("/health")
def health_check():
    if champion_model:
        return {"status": "ok", "champion_model_version": champion_version.version if champion_version else 'unknown'}
    else:
        raise HTTPException(status_code=503, detail="Champion model is not loaded.")
