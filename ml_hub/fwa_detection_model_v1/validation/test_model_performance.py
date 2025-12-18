
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import logging
import sys
from pathlib import Path

# Add project root to path for robust imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        config = load_config('config.json')
        paths = setup_directories(config)
    except FileNotFoundError:
        config = load_config('ml_hub/fwa_detection_model_v1/config.json')
        paths = setup_directories(config)
        
    logging.info("--- FWA Validation: Checking new model performance (AUC) ---")
    
    # Load the held-out test set that was created during the training step
    try:
        X_test = pd.read_csv(Path(paths['data_output_dir']) / 'X_test_fwa.csv')
        y_test = pd.read_csv(Path(paths['data_output_dir']) / 'y_test_fwa.csv')['is_fraud']
    except FileNotFoundError:
        logging.error("Test data not found. Please run the training pipeline first.")
        sys.exit(1)
    
    try:
        # Load the latest version of the model that was just registered
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("fwa-detection-model", stages=["None"])
        if not versions:
            raise Exception("No model versions found for 'fwa-detection-model'.")
            
        latest_version_info = versions[0]
        model_uri = f"models:/fwa-detection-model/{latest_version_info.version}"
        
        logging.info(f"Loading model version {latest_version_info.version} from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        
        # The model's predict_proba output is needed for AUC
        predictions_proba = model.predict(X_test) # With pyfunc, predict returns probabilities for classifiers
        auc = roc_auc_score(y_test, predictions_proba)
        
        logging.info(f"New model AUC on test set: {auc:.4f}")
        
        min_auc = config['validation_thresholds']['min_auc']
        if auc < min_auc:
            logging.error(f"Validation FAILED: Model AUC {auc:.4f} is below the threshold of {min_auc}.")
            sys.exit(1) # Fail the CI/CD job
        else:
            logging.info(f"Validation PASSED: Model AUC {auc:.4f} meets the threshold of {min_auc}.")

    except Exception as e:
        logging.error(f"Failed to load or validate model: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
