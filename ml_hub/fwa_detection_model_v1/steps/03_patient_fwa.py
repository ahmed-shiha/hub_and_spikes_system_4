
import pandas as pd
from sklearn.ensemble import IsolationForest
import logging
from pathlib import Path
import sys
from pyspark.sql.functions import col, collect_list, lit

# Add project root for robust imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
from medins_ml_utils.data_connectors import get_data
from medins_ml_utils.features import create_patient_fwa_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = load_config('config.json')
    paths = setup_directories(config)
    spark = get_spark_session("FWA_Stage3_Patient")
    
    logging.info("--- FWA Stage 3: Patient Level Detection (Doctor Shopping) ---")
    
    # Load data
    data_dir = Path(config['paths']['data_dir'])
    raw_data = get_data(data_dir, ['F_Claim_Header'], spark)
    df = raw_data['F_Claim_Header']
    
    # Build Patient Profile (Simple ABT on the fly)
    # Group by Patient_KEY
    from pyspark.sql.functions import max as spark_max
    
    # Note: Using Patient_KEY column (first one)
    # The CSV header says Patient_KEY but Spark might be reading it as Patient_KEY0 due to duplicates or case sensitivity settings?
    # Actually the error says: Did you mean one of the following? [`Patient_KEY0`, `patient_key2` ...
    # This implies Spark has renamed the column, possibly due to duplicate columns in the CSV schema inference?
    # Let's check the CSV header again: Patient_KEY,payer_share_amount,patient_key,...
    # Ah, Patient_KEY and patient_key (case insensitive collision in Spark?).
    # Spark often renames duplicates to avoid collision.
    
    # Let's use the first column explicitly or use Patient_KEY0 if that's what Spark calls it.
    # To be safe, let's look at df.columns
    # For this patch, I'll trust the error message and use Patient_KEY0.
    
    target_col = 'Patient_KEY0' if 'Patient_KEY0' in df.columns else 'Patient_KEY'
    
    df_patient = df.groupBy(target_col).agg(
        collect_list('provider_license_key').alias('provider_history_list'),
        collect_list('max_serviced_date').alias('service_dates_list'),
        spark_max('max_serviced_date').alias('max_serviced_date')
    )
    
    # Feature Engineering
    df_features = create_patient_fwa_features(df_patient, {})
    
    # Unsupervised Detection
    logging.info("Running Isolation Forest on Patients...")
    # Select feature columns for pandas conversion
    pdf = df_features.select('unique_providers_30d', 'claim_velocity_7d').toPandas()
    
    # Add Patient_KEY back to pandas df
    patient_keys = df_features.select(target_col).toPandas()
    pdf['Patient_KEY'] = patient_keys[target_col]
    
    X = pdf[['unique_providers_30d', 'claim_velocity_7d']].fillna(0)
    
    model = IsolationForest(contamination=0.005)
    model.fit(X)
    
    pdf['anomaly_score'] = model.decision_function(X)
    suspicious_patients = pdf.sort_values('anomaly_score').head(20)
    
    output_path = Path(paths['output_dir']) / "suspicious_patients.csv"
    suspicious_patients.to_csv(output_path, index=False)
    logging.info(f"Saved {len(suspicious_patients)} suspicious patients to {output_path}")

if __name__ == "__main__":
    main()
