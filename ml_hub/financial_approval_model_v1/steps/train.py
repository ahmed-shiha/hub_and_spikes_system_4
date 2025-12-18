
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import mlflow

def run_step():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    dirs = setup_directories(config)
    spark = get_spark_session("FinancialApproval_Train")
    
    # Load Features
    feat_path = dirs['data_output_dir'] / "features_financial.parquet"
    df = spark.read.parquet(str(feat_path))
    
    # Convert to Pandas for Training
    pdf = df.select("plan_utilization_percent", "claim_amount", "provider_network_tier", "is_duplicate_suspect").toPandas()
    
    # Create Dummy Target (Simulation)
    # Reject if utilization > 90% OR out-of-network (tier 2) OR duplicate
    pdf['is_financially_rejected'] = ((pdf['plan_utilization_percent'] > 0.9) | 
                                      (pdf['provider_network_tier'] > 1) | 
                                      (pdf['is_duplicate_suspect'] == 1)).astype(int)
    
    X = pdf[["plan_utilization_percent", "claim_amount", "provider_network_tier", "is_duplicate_suspect"]]
    y = pdf['is_financially_rejected']
    
    # Train
    model = lgb.LGBMClassifier(**config['model_params'])
    model.fit(X, y)
    
    # Save
    model_path = dirs['model_output_dir'] / "financial_model.txt"
    model.booster_.save_model(str(model_path))
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    run_step()
