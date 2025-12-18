
import mlflow
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config
from medins_ml_utils.data_connectors import get_data

logging.basicConfig(level=logging.INFO)

def main():
    config = load_config()
    logging.info("--- Network Value Model: Batch Scoring Providers ---")
    
    tables_to_load = ['F_Claim_Header']
    data = get_data(Path(config['paths']['data_dir']), tables_to_load)
    claims = data['F_Claim_Header']
    
    # Re-create the same features as the training script
    provider_features = claims.groupby('provider_license_key').agg(
        avg_cost_per_claim=('payer_share_amount', 'mean'),
        claim_count=('claim_request_bundle_id', 'nunique')
    ).reset_index()
    np.random.seed(42) # Use same seed for consistent simulation
    provider_features['provider_fraud_score'] = np.random.lognormal(0, 1, len(provider_features)) * provider_features['avg_cost_per_claim'] * 0.0001
    provider_features = provider_features.fillna(0)

    features_to_cluster = ['avg_cost_per_claim', 'claim_count', 'provider_fraud_score']
    X = provider_features[features_to_cluster]
    X_scaled = StandardScaler().fit_transform(X)

    try:
        model_uri = "models:/network-value-model/Production"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        
        client = mlflow.tracking.MlflowClient()
        prod_version = client.get_latest_versions("network-value-model", stages=["Production"])[0]
        artifact_path = client.download_artifacts(prod_version.run_id, "cluster_labels.json")
        with open(artifact_path, 'r') as f:
            cluster_labels_map = {int(k): v for k, v in json.load(f).items()}

        labels = model.predict(X_scaled)
        provider_features['value_tier_id'] = labels
        provider_features['value_tier_name'] = provider_features['value_tier_id'].map(cluster_labels_map)
        
        output_path = Path("output/provider_value_tiers.csv")
        output_path.parent.mkdir(exist_ok=True)
        provider_features.to_csv(output_path, index=False)
        logging.info(f"Provider value tiers saved to {output_path}")

    except Exception as e:
        logging.error(f"Failed to score providers: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
