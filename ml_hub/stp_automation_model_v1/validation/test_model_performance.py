
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
import logging
import sys
from pathlib import Path
import json

# Add project root to path for robust imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        config = load_config('config.json')
        paths = setup_directories(config)
    except FileNotFoundError:
        config = load_config('ml_hub/stp_automation_model_v1/config.json')
        paths = setup_directories(config)
        
    logging.info("--- STP Validation: Checking model precision against business thresholds ---")
    
    # Load the held-out test set that was created during the training step
    try:
        X_test = pd.read_csv(Path(paths['data_output_dir']) / 'X_test_stp.csv')
        y_test_encoded = pd.read_csv(Path(paths['data_output_dir']) / 'y_test_stp.csv').iloc[:, 0]
    except FileNotFoundError:
        logging.error("Test data not found. Please run the training pipeline first.")
        sys.exit(1)
    
    try:
        client = mlflow.tracking.MlflowClient()
        latest_version_info = client.get_latest_versions("stp-automation-model", stages=["None"])[0]
        model_uri = f"models:/stp-automation-model/{latest_version_info.version}"
        logging.info(f"Loading model version {latest_version_info.version} from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        
        artifact_path = client.download_artifacts(latest_version_info.run_id, "class_mapping.json")
        with open(artifact_path, 'r') as f:
            class_mapping = {int(k): v for k, v in json.load(f).items()}
        
        label_to_int = {v: k for k, v in class_mapping.items()}

        y_pred_proba = model.predict(X_test)
        y_pred_class_indices = np.argmax(y_pred_proba, axis=1)

        approve_precision = precision_score(y_test_encoded, y_pred_class_indices, labels=[label_to_int.get('APPROVE', -1)], average='micro', zero_division=0)
        deny_precision = precision_score(y_test_encoded, y_pred_class_indices, labels=[label_to_int.get('DENY', -1)], average='micro', zero_division=0)
        
        logging.info(f"New model Precision on 'APPROVE' class: {approve_precision:.4f}")
        logging.info(f"New model Precision on 'DENY' class: {deny_precision:.4f}")

        thresholds = config['validation_thresholds']
        approve_threshold = thresholds['min_approve_precision']
        deny_threshold = thresholds['min_deny_precision']
        
        passed = True
        if approve_precision < approve_threshold:
            logging.error(f"Validation FAILED: Approve precision {approve_precision:.4f} is below the required threshold of {approve_threshold}.")
            passed = False
        if deny_precision < deny_threshold:
            logging.error(f"Validation FAILED: Deny precision {deny_precision:.4f} is below the required threshold of {deny_threshold}.")
            passed = False

        if passed:
            logging.info("Validation PASSED: Model precision meets all business thresholds.")
        else:
            sys.exit(1) # Fail the CI/CD pipeline

    except Exception as e:
        logging.error(f"Failed to load or validate model: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
