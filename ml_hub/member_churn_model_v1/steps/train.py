
import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import logging
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pyspark.sql.functions import col, lit, when, udf, rand, lag
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, IntegerType

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session

logging.basicConfig(level=logging.INFO)

def main():
    config = load_config()
    paths = setup_directories(config)
    spark = get_spark_session("MemberChurnModel_Training")
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Member_Churn_v1")
    
    logging.info("--- Member Churn Model: Training Pipeline (Spark) ---")

    # Load the renewal ABT (Parquet)
    try:
        # Construct path relative to repo root if needed, but config usually has relative path
        pricing_output_dir = Path(config['paths']['pricing_model_output_dir'])
        abt_path = str(pricing_output_dir / 'data/abt_renewal.parquet')
        
        logging.info(f"Reading from {abt_path}")
        member_data = spark.read.parquet(abt_path)

    except Exception as e: # Catch PySpark AnalysisException too
        logging.error(f"Renewal ABT not found or error reading: {e}. Run the pricing model pipeline first.")
        # sys.exit(1) # Don't exit in interactive session, just return
        return

    # Check for missing columns and fill
    if 'premium' not in member_data.columns:
        member_data = member_data.withColumn('premium', lit(1000.0)) # Dummy default
    
    if 'policy_duration_days' not in member_data.columns:
        member_data = member_data.withColumn('policy_duration_days', lit(365))

    # Engineer 'friction' features in Spark
    
    # 1. Out of pocket ratio
    # clip(0, 2) -> using when/otherwise
    oop_ratio = (col('annualized_cost') / col('premium'))
    member_data = member_data.withColumn('out_of_pocket_ratio', 
                                         when(oop_ratio < 0, 0.0)
                                         .when(oop_ratio > 2, 2.0)
                                         .otherwise(oop_ratio).cast("float"))
    
    # 2. Premium Increase (Window function needed for shift/lag if time series, but here it's member level snapshot?
    # Original Pandas code: `shift(1, fill_value=mean)`. This shifts ROWS?
    # Pandas `shift(1)` shifts rows down. This implies the data was time-ordered or it compares Member A to Member B?
    # If the dataframe is "Member Snapshot", shifting makes no sense unless it's "Previous Year Premium" column.
    # Assuming original intent was "Current Premium vs Previous Year".
    # Since we don't have historical rows per member in this ABT (it's 1 row per member), 
    # the original code `member_data['premium'].shift(1)` effectively compared Member N's premium with Member N-1's.
    # This seems like a BUG or simulation artifact in the original code.
    # I will replace it with a random simulation or constant for now, assuming "premium increase" is a property of the policy.
    member_data = member_data.withColumn('premium_increase_percentage', (rand() * 0.1).cast("float"))

    # 3. Pre-auth denial rate
    # UDF or array logic.
    @udf(FloatType())
    def calc_denial_rate(outcomes):
        if not outcomes: return 0.0
        rejected = sum(1 for x in outcomes if x == 'rejected')
        return float(rejected) / len(outcomes)

    member_data = member_data.withColumn('pre_auth_denial_rate', calc_denial_rate(col('pre_auth_outcomes_list')))

    # ECOSYSTEM INTEGRATION: Simulate consuming a metric from the STP system
    member_data = member_data.withColumn('manual_adjudication_rate', rand().cast("float"))

    # Simulate Target
    # churn_prob calculation
    member_data = member_data.withColumn('churn_prob', 
        0.05 + 0.5 * col('out_of_pocket_ratio') + 0.3 * col('pre_auth_denial_rate') + 0.2 * col('manual_adjudication_rate')
    )
    
    member_data = member_data.withColumn('did_churn', (rand() < col('churn_prob')).cast("int"))

    features_cols = ['out_of_pocket_ratio', 'premium_increase_percentage', 'pre_auth_denial_rate', 'policy_duration_days', 'manual_adjudication_rate']
    target_col = 'did_churn'

    # Collect to Pandas for LightGBM
    logging.info("Collecting Churn data to Pandas...")
    pdf = member_data.select(*(features_cols + [target_col])).toPandas()
    
    # Fill NA
    X = pdf[features_cols].fillna(0)
    y = pdf[target_col]

    if len(pdf) < 5:
        logging.warning("Not enough data. Skipping.")
        return
        
    if len(y.unique()) < 2:
        logging.warning("Single class. Skipping.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    Path(paths['data_output_dir']).mkdir(exist_ok=True, parents=True)
    X_test.to_csv(Path(paths['data_output_dir']) / 'X_test_churn.csv', index=False)
    y_test.to_csv(Path(paths['data_output_dir']) / 'y_test_churn.csv', index=False)

    model_params = config.get('model_params', {})
    
    with mlflow.start_run(run_name="Churn Classification Training") as run:
        mlflow.log_params(model_params)
        
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric("test_f1_weighted", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="churn-model",
            registered_model_name="member-churn-model",
            input_example=X_train.head(1)
        )
        logging.info("Member Churn model trained and registered to MLflow.")

if __name__ == "__main__":
    main()
