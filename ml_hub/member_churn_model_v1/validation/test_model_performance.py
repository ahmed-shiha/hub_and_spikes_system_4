
import mlflow
import pandas as pd
from sklearn.metrics import f1_score
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories

logging.basicConfig(level=logging.INFO)

def main():
    try:
        config = load_config('config.json')
        paths = setup_directories(config)
    except FileNotFoundError:
        config = load_config('ml_hub/member_churn_model_v1/config.json')
        paths = setup_directories(config)
        
    logging.info("--- Member Churn Validation: Checking Model F1-Score ---")

    try:
        X_test = pd.read_csv(Path(paths['data_output_dir']) / 'X_test_churn.csv')
        y_test = pd.read_csv(Path(paths['data_output_dir']) / 'y_test_churn.csv').iloc[:, 0]
    except FileNotFoundError:
        logging.error("Test data not found. Please run the training pipeline first.")
        sys.exit(1)
    
    try:
        latest_version_info = mlflow.tracking.MlflowClient().get_latest_versions("member-churn-model", stages=["None"])[0]
        model_uri = f"models:/member-churn-model/{latest_version_info.version}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average='weighted')
        
        logging.info(f"New model F1-Score on test set: {score:.4f}")
        
        min_score = config['validation_thresholds']['min_f1_score']
        if score < min_score:
            logging.error(f"Validation FAILED: F1-Score {score:.4f} is below threshold of {min_score}.")
            sys.exit(1)
        else:
            logging.info(f"Validation PASSED: F1-Score {score:.4f} meets threshold.")

    except Exception as e:
        logging.error(f"Failed to load or validate model: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
