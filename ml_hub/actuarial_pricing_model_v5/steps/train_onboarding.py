
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
    spark = get_spark_session("Pricing_Train_Onboarding")
    
    # Load Features
    feat_path = dirs['data_output_dir'] / "features_pricing_onboarding.parquet"
    df = spark.read.parquet(str(feat_path))
    
    # Convert to Pandas
    pdf = df.select("age", "self_reported_chronic_score", "future_annual_cost").toPandas()
    
    X = pdf[["age", "self_reported_chronic_score"]]
    y = pdf['future_annual_cost']
    
    # Train
    # We use a simpler model for Onboarding as we have fewer features
    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=100)
    model.fit(X, y)
    
    # Save
    model_path = dirs['model_output_dir'] / "pricing_onboarding_model.txt"
    model.booster_.save_model(str(model_path))
    logging.info(f"Onboarding Model saved to {model_path}")

if __name__ == "__main__":
    run_step()
