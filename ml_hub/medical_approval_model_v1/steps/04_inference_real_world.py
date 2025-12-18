
from medins_ml_utils.utils import load_config, setup_directories, load_artifacts
from medins_ml_utils.real_world_connector import parse_user_json, flatten_claim_dump
import pandas as pd
import logging
import lightgbm as lgb
from pathlib import Path

def run_step():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    dirs = setup_directories(config)
    
    logging.info("--- Medical Model: Real World Inference Simulation ---")
    
    # 1. Load Real World Data
    real_data_path = Path(dirs['data_dir']) / "real_claim_data/TB Medical Claim Request.csv"
    if not real_data_path.exists():
        logging.warning("Real claim data file not found. Skipping.")
        return

    # 2. Parse User Context (Simulated as we don't have the JSON file separate, usually it comes with the request)
    # Using the snippet provided in the prompt
    user_json_str = '{"name":"EMP-443","gender":"Male","birth_date":"1998-01-01","had_chronic_diseases":0}'
    user_features = parse_user_json(user_json_str)
    logging.info(f"User Context: {user_features}")
    
    # 3. Flatten Claims
    # Note: The provided CSV is very complex. The flatten_claim_dump is a simplification.
    # In a real pipeline, we'd use a robust ETL tool.
    # For now, we simulate the dataframe structure that matches our model input
    
    # Mocking the result of flattening for demonstration of the Pipeline flow
    inference_data = [
        {"procedure_code": "PROC_CHECKUP", "diagnosis_code": "E11", "activity_type": "Service"}, # Match
        {"procedure_code": "PROC_C_SECTION", "diagnosis_code": "O82", "activity_type": "Service"} # Gender Mismatch check
    ]
    df = pd.DataFrame(inference_data)
    
    # Add User Features to every row
    df['Gender'] = user_features['gender']
    df['age'] = user_features['age']
    
    logging.info(f"Inference Batch: {len(df)} items")
    
    # 4. Feature Engineering (On the Fly)
    # We must replicate the Logic from features.py but for Pandas (Online Inference)
    # Ideally, features.py handles both Spark and Pandas, or we use the 'online_features' module.
    # For v1, we reimplement simple lookups.
    
    artifacts = load_artifacts(dirs['model_output_dir'])
    soc = artifacts.get('standard_of_care', {})
    gender_rules = artifacts.get('gender_restricted_procedures', {})
    
    def is_soc_compliant(row):
        allowed = soc.get(row['diagnosis_code'], [])
        return 1 if row['procedure_code'] in allowed else 0 # Default 0 if strict, 1 if loose

    def is_gender_compliant(row):
        allowed = gender_rules.get(row['procedure_code'])
        if not allowed: return 1
        return 1 if row['Gender'] in allowed else 0

    df['is_soc_compliant'] = df.apply(is_soc_compliant, axis=1)
    df['is_gender_compliant'] = df.apply(is_gender_compliant, axis=1)
    
    # 5. Prediction
    model_path = dirs['model_output_dir'] / "medical_model.txt"
    model = lgb.Booster(model_file=str(model_path))
    
    X = df[["is_soc_compliant", "is_gender_compliant"]]
    df['medical_necessity_prob'] = model.predict(X)
    
    logging.info("--- Inference Results ---")
    print(df[['procedure_code', 'diagnosis_code', 'Gender', 'medical_necessity_prob']])

if __name__ == "__main__":
    run_step()
