
from medins_ml_utils.utils import load_config, setup_directories, load_artifacts, get_spark_session
# Note: we need to import the new function. Ideally we'd expose it in __init__ but for now we import from features
from medins_ml_utils.features import create_financial_approval_features
import logging

def run_step():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    dirs = setup_directories(config)
    spark = get_spark_session("FinancialApproval_Features")
    
    # Load ABT
    abt_path = dirs['data_output_dir'] / "abt_financial.parquet"
    df = spark.read.parquet(str(abt_path))
    
    # Load Artifacts
    artifacts = load_artifacts(dirs['model_output_dir'])
    
    # Feature Engineering
    df_features = create_financial_approval_features(df, artifacts)
    
    # Output
    output_path = dirs['data_output_dir'] / "features_financial.parquet"
    df_features.write.mode("overwrite").parquet(str(output_path))
    logging.info(f"Features saved to {output_path}")

if __name__ == "__main__":
    run_step()
