
from medins_ml_utils.utils import load_config, setup_directories, load_artifacts, get_spark_session
from medins_ml_utils.features import create_medical_approval_features
import logging

def run_step():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    dirs = setup_directories(config)
    spark = get_spark_session("MedicalApproval_Features")
    
    abt_path = dirs['data_output_dir'] / "abt_medical.parquet"
    df = spark.read.parquet(str(abt_path))
    
    artifacts = load_artifacts(dirs['model_output_dir'])
    
    df_features = create_medical_approval_features(df, artifacts)
    
    output_path = dirs['data_output_dir'] / "features_medical.parquet"
    df_features.write.mode("overwrite").parquet(str(output_path))
    logging.info(f"Medical Features saved to {output_path}")

if __name__ == "__main__":
    run_step()
