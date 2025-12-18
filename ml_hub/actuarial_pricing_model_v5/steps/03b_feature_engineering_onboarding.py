
from medins_ml_utils.utils import load_config, setup_directories, load_artifacts, get_spark_session
from medins_ml_utils.features import create_onboarding_features
import logging

def run_step():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    dirs = setup_directories(config)
    spark = get_spark_session("Pricing_Features_Onboarding")
    
    # Load ABT
    abt_path = dirs['data_output_dir'] / "abt_pricing_onboarding.parquet"
    df = spark.read.parquet(str(abt_path))
    
    # Load Artifacts (Global Disease Weights from Step 01)
    artifacts = load_artifacts(dirs['model_output_dir'])
    
    # Feature Engineering
    # This uses the "Self-Reported" logic in the Hub
    df_features = create_onboarding_features(df, artifacts)
    
    # Output
    output_path = dirs['data_output_dir'] / "features_pricing_onboarding.parquet"
    df_features.write.mode("overwrite").parquet(str(output_path))
    logging.info(f"Onboarding Features saved to {output_path}")

if __name__ == "__main__":
    run_step()
