
import mlflow
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import logging
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories

logging.basicConfig(level=logging.INFO)

def main():
    try:
        config = load_config('config.json')
        paths = setup_directories(config)
    except FileNotFoundError:
        config = load_config('ml_hub/network_value_model_v1/config.json')
        paths = setup_directories(config)
        
    logging.info("--- Network Value Validation: Checking Cluster Separation ---")

    # Load the scaled feature data that was used for training
    try:
        X_val = pd.read_csv(Path(paths['data_output_dir']) / 'network_features_scaled.csv')
    except FileNotFoundError:
        logging.error("Scaled feature data not found. Please run the training pipeline first.")
        sys.exit(1)
    
    try:
        latest_version_info = mlflow.tracking.MlflowClient().get_latest_versions("network-value-model", stages=["None"])[0]
        model_uri = f"models:/network-value-model/{latest_version_info.version}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        
        kmeans_model = model._model_impl 
        labels = kmeans_model.labels_

        if len(labels) != len(X_val):
             labels = kmeans_model.predict(X_val)

        score = silhouette_score(X_val, labels)
        
        logging.info(f"New model Silhouette Score: {score:.4f}")
        
        min_score = config['validation_thresholds']['min_silhouette_score']
        if score < min_score:
            logging.error(f"Validation FAILED: Silhouette Score {score:.4f} is below threshold of {min_score}.")
            sys.exit(1)
        else:
            logging.info(f"Validation PASSED: Silhouette Score {score:.4f} meets threshold.")

    except Exception as e:
        logging.error(f"Failed to load or validate model: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
