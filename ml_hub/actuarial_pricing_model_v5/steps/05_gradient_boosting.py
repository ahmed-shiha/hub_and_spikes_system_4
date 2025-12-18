
import sys
from pathlib import Path
import logging
import pandas as pd
import lightgbm as lgb
import joblib
import mlflow
import os
from sklearn.metrics import mean_absolute_error
import numpy as np

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_lgbm_model(model_type: str, config: dict, paths: dict, parent_run_id: str, spark):
    """Trains a specific LightGBM model variant and logs it to MLflow under a parent run."""
    logging.info(f"--- Training LightGBM for '{model_type}' model ---")
    
    abt_featured_path = str(Path(paths['data_output_dir']) / f'abt_{model_type}_featured.parquet')
    if not Path(abt_featured_path).exists():
        logging.warning(f"ABT file not found for model type '{model_type}'. Skipping.")
        return

    # Read with Spark
    abt_featured_spark = spark.read.parquet(abt_featured_path)
    
    # Check volume. If > X GB, we might need distributed training.
    # For this plan, we assume aggregated feature table fits in memory of the training node (common pattern).
    # If not, we would use SynapseML LightGBM.
    # Strategy: Filter Train/Test using Spark, then collect to Pandas.
    
    train_ids_df = spark.read.option("header", "true").csv(str(Path(paths['data_output_dir']) / 'train_member_ids.csv'))
    test_ids_df = spark.read.option("header", "true").csv(str(Path(paths['data_output_dir']) / 'test_member_ids.csv'))
    
    # Broadcast join for IDs if they are small relative to data, but standard join is fine.
    train_df_spark = abt_featured_spark.join(train_ids_df, on='Patient_KEY', how='inner')
    test_df_spark = abt_featured_spark.join(test_ids_df, on='Patient_KEY', how='inner')
    
    logging.info("Collecting training data to Pandas (Driver Memory)...")
    # This is the bottleneck if data is huge.
    # Mitigation: Downsample in Spark if needed.
    # train_df_spark = train_df_spark.sample(fraction=0.1) 
    
    train_df = train_df_spark.toPandas()
    test_df = test_df_spark.toPandas()
    
    # Ensure numeric types (Spark toPandas might keep decimals/objects)
    # Using 'ignore' is deprecated, so we iterate or use error handling.
    # Also, LightGBM needs numeric or bool.
    # We should convert object columns to 'category' type for LightGBM.
    
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            # Check if it's a list (stringified or actual list)
            # If actual list, we must drop it or expand. We previously decided to drop lists.
            # But here we might have string columns like 'gender', 'city' which are categorical.
            
            # Simple heuristic: if it looks like a list, drop it. Else make it categorical.
            sample = train_df[col].dropna().iloc[0] if len(train_df[col].dropna()) > 0 else None
            if isinstance(sample, (list, np.ndarray)):
                train_df = train_df.drop(columns=[col])
                test_df = test_df.drop(columns=[col])
            else:
                train_df[col] = train_df[col].astype('category')
                test_df[col] = test_df[col].astype('category')
        else:
             # Force numeric for non-objects to be safe (e.g. decimal objects)
             train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
             test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    if 'annualized_cost' not in train_df.columns:
         logging.error("Target 'annualized_cost' missing.")
         return

    y_train = train_df['annualized_cost']
    y_test = test_df['annualized_cost']
    
    cols_to_drop_config = config.get('columns_to_drop', {})
    features_to_drop = []
    for key, cols in cols_to_drop_config.items():
        features_to_drop.extend(cols)
    
    X_train = train_df.drop(columns=features_to_drop, errors='ignore')
    X_test = test_df.drop(columns=features_to_drop, errors='ignore')
    
    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # Check if empty
    if X_train.empty:
        logging.warning(f"Training data empty for {model_type}. Skipping.")
        return
        
    if X_test.empty:
        logging.warning(f"Test data empty for {model_type}. Skipping evaluation.")
        # Proceed with train but skip eval metrics
    
    lgbm_params = config['lgbm_params']

    # MLflow Nested Run
    with mlflow.start_run(run_name=f"Train_{model_type}", nested=True):
        mlflow.log_params(lgbm_params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("training_sample_size", len(X_train))
        
        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_train, y_train)

        if not X_test.empty:
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mlflow.log_metric(f"test_mae", mae)
            logging.info(f"MAE for '{model_type}': {mae:,.2f} SAR")
            
            # Save test data for validation stage (CSV is fine for small validation sets)
            y_test.to_csv(Path(paths['data_output_dir']) / f'y_test_{model_type}.csv', index=False, header=True)
            pd.DataFrame(y_pred, columns=['y_pred']).to_csv(Path(paths['data_output_dir']) / f'y_pred_{model_type}.csv', index=False)

        model_name = f"pricing-{model_type.replace('_', '-')}"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            registered_model_name=model_name,
            input_example=X_train.head(1)
        )
        
        columns_path = Path(paths['model_output_dir']) / f'lgbm_model_columns_{model_type}.pkl'
        joblib.dump(X_train.columns.tolist(), columns_path)
        mlflow.log_artifact(str(columns_path))


def main():
    config = load_config()
    paths = setup_directories(config)
    spark = get_spark_session("PricingModel_Stage5_Training")
    
    logging.info("--- Pricing Model: Stage 5 - Gradient Boosting Training & Registration ---")
    
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("Pricing_Model_v5")
    
    with mlflow.start_run(run_name="Pricing Model Training Run") as parent_run:
        parent_run_id = parent_run.info.run_id
        logging.info(f"MLflow Parent Run ID: {parent_run_id}")
        
        mlflow.log_param("snapshot_date", config['snapshot_date'])
        mlflow.set_tag("project", "actuarial_pricing_model_v5")

        model_types = ['onboarding', 'renewal_vice_champion', 'renewal_champion']
        
        for model_type in model_types:
            train_lgbm_model(model_type, config, paths, parent_run_id, spark)
            
    logging.info("--- Stage 5 Complete ---")

if __name__ == "__main__":
    main()
