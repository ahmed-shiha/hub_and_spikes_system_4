
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
import lightgbm as lgb
import pandas as pd
import logging

def run_step():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    dirs = setup_directories(config)
    spark = get_spark_session("MedicalApproval_Train")
    
    feat_path = dirs['data_output_dir'] / "features_medical.parquet"
    df = spark.read.parquet(str(feat_path))
    
    pdf = df.select("is_soc_compliant", "is_gender_compliant").toPandas()
    
    # Ground Truth: If it violates rules, it's NOT necessary
    pdf['is_medically_necessary'] = ((pdf['is_soc_compliant'] == 1) & (pdf['is_gender_compliant'] == 1)).astype(int)
    
    X = pdf[["is_soc_compliant", "is_gender_compliant"]]
    y = pdf['is_medically_necessary']
    
    model = lgb.LGBMClassifier(**config['model_params'])
    model.fit(X, y)
    
    model_path = dirs['model_output_dir'] / "medical_model.txt"
    model.booster_.save_model(str(model_path))
    logging.info(f"Medical Model saved to {model_path}")

if __name__ == "__main__":
    run_step()
