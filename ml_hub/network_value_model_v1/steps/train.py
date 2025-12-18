
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import mlflow
import logging
from pathlib import Path
import json
import sys
from pyspark.sql.functions import col, avg, countDistinct, rand, lit

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
from medins_ml_utils.data_connectors import get_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = load_config('config.json')
    paths = setup_directories(config)
    spark = get_spark_session("NetworkValueModel_Training")
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Network_Value_v1")
    
    logging.info("--- Network Value Model: Training Pipeline (Spark) ---")

    # Load real claims data
    tables_to_load = ['F_Claim_Header']
    data = get_data(Path(config['paths']['data_dir']), tables_to_load, spark)
    claims = data['F_Claim_Header']
    
    # Engineer provider-level features using Spark
    provider_features_spark = claims.groupBy('provider_license_key').agg(
        avg('payer_share_amount').alias('avg_cost_per_claim'),
        countDistinct('claim_request_bundle_id').alias('claim_count')
    ).na.fill(0)

    # Collect to Pandas for Sklearn KMeans (Clustering usually on summarized data)
    logging.info("Collecting provider features to Pandas...")
    provider_features = provider_features_spark.toPandas()
    
    # Ensure numeric types
    provider_features['avg_cost_per_claim'] = pd.to_numeric(provider_features['avg_cost_per_claim'])
    provider_features['claim_count'] = pd.to_numeric(provider_features['claim_count'])

    # ECOSYSTEM INTEGRATION: Simulate consuming fraud scores from the FWA system
    if len(provider_features) > 0:
        np.random.seed(42)
        provider_features['provider_fraud_score'] = np.random.lognormal(0, 1, len(provider_features)) * provider_features['avg_cost_per_claim'] * 0.0001
    else:
        provider_features['provider_fraud_score'] = 0.0
        
    provider_features = provider_features.fillna(0)

    features_to_cluster = [
        'avg_cost_per_claim', 
        'claim_count',
        'provider_fraud_score'
    ]
    
    if len(provider_features) < 5:
        logging.warning("Not enough provider data for clustering. Skipping.")
        return

    X = provider_features[features_to_cluster]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save validation data
    Path(paths['data_output_dir']).mkdir(exist_ok=True, parents=True)
    pd.DataFrame(X_scaled, columns=features_to_cluster).to_csv(Path(paths['data_output_dir']) / 'network_features_scaled.csv', index=False)

    model_params = config.get('model_params', {})
    
    with mlflow.start_run(run_name="KMeans Clustering") as run:
        mlflow.log_params(model_params)
        
        model = KMeans(**model_params)
        model.fit(X_scaled)
        
        provider_features['cluster_label'] = model.labels_
        
        cluster_labels_path = Path(paths['output_dir']) / "cluster_labels.json"
        with open(cluster_labels_path, 'w') as f:
            json.dump(config.get('cluster_labels', {}), f)
        mlflow.log_artifact(str(cluster_labels_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="network-value-model",
            registered_model_name="network-value-model",
            input_example=pd.DataFrame(X_scaled, columns=features_to_cluster).head(1)
        )
        logging.info("Network Value model trained and registered to MLflow.")

if __name__ == "__main__":
    main()
