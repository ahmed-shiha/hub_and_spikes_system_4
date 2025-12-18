
import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import json
import sys
from pyspark.sql.functions import col, lit, when, udf, rand
from pyspark.sql.types import IntegerType, StringType

# Add project root to path for robust imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
from medins_ml_utils.data_connectors import get_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = load_config()
    paths = setup_directories(config)
    spark = get_spark_session("STP_Training")
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("STP_Automation_v1")
    
    logging.info("--- STP Training Pipeline: Building Data and Training Model (Spark) ---")
    
    # --- 1. Build Labeled Training Data ---
    # Load real claims data to apply business rules
    tables_to_load = ['F_Claim_Header']
    data = get_data(Path(config['paths']['data_dir']), tables_to_load, spark)

    claims_df = data['F_Claim_Header']
    
    # Check if 'net_amount' exists, if not use 'payer_share_amount'
    if 'net_amount' not in claims_df.columns:
        if 'payer_share_amount' in claims_df.columns:
            claims_df = claims_df.withColumn('net_amount', col('payer_share_amount'))
        else:
            logging.error("net_amount or payer_share_amount missing.")
            return

    # Create features for the labeling logic
    # 'pre_auth_ref' check
    if 'pre_auth_ref' in claims_df.columns:
        claims_df = claims_df.withColumn('has_pre_auth', when(col('pre_auth_ref').isNotNull(), 1).otherwise(0))
    else:
        claims_df = claims_df.withColumn('has_pre_auth', lit(0)) # Default if missing
    
    # Simulate 'service_is_covered' and 'is_complex_diag' for labeling
    # In Spark, we use random generator or deterministic logic if needed.
    # rand() < 0.9 means 90% covered.
    claims_df = claims_df.withColumn('service_is_covered', when(rand() < 0.9, 1).otherwise(0))
    claims_df = claims_df.withColumn('is_complex_diag', when(rand() < 0.2, 1).otherwise(0))

    # Apply business rules to create the target variable (ground truth)
    # Using 'when' logic instead of UDF for performance
    
    claims_df = claims_df.withColumn(
        'adjudication_target',
        when((col('has_pre_auth') == 1) & (col('net_amount') < 500) & (col('service_is_covered') == 1), 'APPROVE')
        .when(col('service_is_covered') == 0, 'DENY')
        .when((col('is_complex_diag') == 1) | (col('net_amount') > 10000), 'PEND')
        .otherwise('PEND')
    )
    
    # --- 2. Train Multi-Class Classifier ---
    features_cols = ['net_amount', 'has_pre_auth', 'service_is_covered', 'is_complex_diag']
    target_col = 'adjudication_target'
    
    # Collect to Pandas for LightGBM (assuming aggregated/filtered data is manageable)
    logging.info("Collecting training data to Pandas...")
    pdf = claims_df.select(*(features_cols + [target_col])).toPandas()
    
    # Ensure numeric
    for c in features_cols:
        pdf[c] = pd.to_numeric(pdf[c], errors='coerce').fillna(0)

    X = pdf[features_cols]
    y = pdf[target_col]

    y_encoded = y.astype('category').cat.codes
    class_mapping = {i: category for i, category in enumerate(y.astype('category').cat.categories)}

    if len(pdf) < 5:
        logging.warning("Not enough data to split. Skipping.")
        return

    # Check class balance
    if len(y.unique()) < 2:
        logging.warning("Only 1 class found. Skipping.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Save test set for validation stage
    Path(paths['data_output_dir']).mkdir(exist_ok=True, parents=True)
    X_test.to_csv(Path(paths['data_output_dir']) / 'X_test_stp.csv', index=False)
    y_test.to_csv(Path(paths['data_output_dir']) / 'y_test_stp.csv', index=False)

    model_params = config.get('model_params', {})
    
    with mlflow.start_run(run_name="STP Multi-Class Training") as run:
        mlflow.log_params(model_params)
        mlflow.log_metric("training_sample_size", len(X_train))
        
        class_mapping_path = Path(paths['output_dir']) / "class_mapping.json"
        with open(class_mapping_path, 'w') as f:
            json.dump(class_mapping, f)
        mlflow.log_artifact(str(class_mapping_path))
        
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_weighted", f1)
        logging.info(f"Test Set Accuracy: {accuracy:.4f}")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="stp-model",
            registered_model_name="stp-automation-model",
            input_example=X_train.head(1),
            signature=mlflow.models.infer_signature(X_train, model.predict_proba(X_train))
        )
        logging.info("Supervised STP model trained and registered to MLflow as 'stp-automation-model'")

if __name__ == "__main__":
    main()
