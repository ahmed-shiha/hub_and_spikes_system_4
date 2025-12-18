
import pandas as pd
from sklearn.ensemble import IsolationForest
import logging
from pathlib import Path
import numpy as np
import sys
from pyspark.sql.functions import col, avg, stddev, countDistinct, when, dayofweek, lit
from pyspark.sql.types import FloatType

# Add project root for robust imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
from medins_ml_utils.data_connectors import get_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = load_config('config.json')
    paths = setup_directories(config)
    spark = get_spark_session("FWA_Stage1_Unsupervised")
    
    logging.info("--- FWA Stage 1: Generating Anomaly Scores for Investigator Worklist (Spark) ---")
    
    # Load data
    tables_to_load = config.get('tables_to_load', ['F_Claim_Header'])
    data_dir = Path(config['paths']['data_dir'])
    raw_data = get_data(data_dir, tables_to_load, spark)
    raw_claims = raw_data['F_Claim_Header']
    
    # --- Feature Engineering at Provider Level (Spark) ---
    # raw_claims['service_date'] = pd.to_datetime(raw_claims['max_serviced_date'], errors='coerce')
    # Spark implicitly handles dates if schema is inferred or we cast.
    
    # Provider Aggregation
    # avg_cost, claim_count, std_cost
    
    # Weekend billing: SA weekend is Fri/Sat or Thu/Fri depending on logic.
    # Original: x.dt.dayofweek >= 4. (0=Mon, 4=Fri, 5=Sat, 6=Sun).
    # So >=4 covers Fri, Sat, Sun.
    # Spark dayofweek: 1=Sun, 2=Mon... 7=Sat.
    # We want equivalent of Fri, Sat, Sun.
    # Fri=6, Sat=7, Sun=1.
    
    # High cost ratio:
    high_value_threshold = config.get('business_logic', {}).get('large_claim_threshold', 5000)
    
    # We calculate ratios by summing booleans (casted to int) and dividing by count.
    
    provider_features = raw_claims.groupBy('provider_license_key').agg(
        avg('payer_share_amount').alias('avg_cost_per_claim'),
        countDistinct('claim_request_bundle_id').alias('claim_count'),
        stddev('payer_share_amount').alias('std_cost_per_claim'),
        
        # weekend_billing_ratio
        # Spark dayofweek(col): Sun=1, Mon=2 ... Fri=6, Sat=7.
        # Original: >=4 (Fri, Sat, Sun).
        # So we check if dayofweek is 6, 7, or 1.
        (avg(when(dayofweek(col('max_serviced_date')).isin(1, 6, 7), 1).otherwise(0))).alias('weekend_billing_ratio'),
        
        # high_cost_claim_ratio
        (avg(when(col('payer_share_amount') > high_value_threshold, 1).otherwise(0))).alias('high_cost_claim_ratio')
    ).na.fill(0)
    
    # IsolationForest is sklearn. We need to collect to Pandas.
    # Assuming provider summary is small enough (thousands of providers vs millions of claims).
    logging.info("Collecting provider features to driver...")
    pdf = provider_features.toPandas()
    
    features_to_cluster = [
        'avg_cost_per_claim', 
        'claim_count',
        'std_cost_per_claim',
        'weekend_billing_ratio', 
        'high_cost_claim_ratio'
    ]
    X = pdf[features_to_cluster]
    
    model_params = config.get('unsupervised_model_params', {})
    model = IsolationForest(**model_params)
    model.fit(X)
    
    pdf['anomaly_score'] = model.decision_function(X)
    # Lower score = more anomalous
    worklist = pdf.sort_values('anomaly_score').head(20)
    
    output_path = Path(paths['output_dir']) / "fwa_worklist.csv"
    worklist.to_csv(output_path, index=False)
    logging.info(f"Worklist with {len(worklist)} anomalous providers saved to {output_path}")
    
    # --- Enhancement: Save the Unsupervised Model (The Fingerprint) ---
    # We save this isolation forest. It has learned the "Normal Behavior" of the General Market.
    # When we launch our new company, we load this model and score new providers against it immediately.
    import joblib
    model_output_path = Path(paths['model_output_dir']) / "fwa_isolation_forest.pkl"
    joblib.dump(model, model_output_path)
    logging.info(f"Saved FWA Unsupervised Model to {model_output_path}")

if __name__ == "__main__":
    main()
