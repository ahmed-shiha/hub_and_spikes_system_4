
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
from medins_ml_utils.data_connectors import get_data
from pyspark.sql.functions import col, rand, lit, array
import logging

def run_step():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    dirs = setup_directories(config)
    spark = get_spark_session("MedicalApproval_ABT")
    
    # Load Header and Patient
    data = get_data(dirs['data_dir'], ["F_Claim_Header", "D_Patient"], spark)
    df = data["F_Claim_Header"]
    df_pat = data["D_Patient"]
    
    # Check column names
    pat_col = 'Patient_KEY0' if 'Patient_KEY0' in df.columns else 'Patient_KEY'
    
    # Join to get Age/Gender
    df_abt = df.join(df_pat, df[pat_col] == df_pat.Patient_KEY, "left")
    
    # Enhance with Raw Data if available (Service Type)
    # Ideally join with d_product_or_service on service code
    # For now, we simulate 'activity_type' which is crucial for medical logic
    
    # Mocking activity type based on randomness for now, but in prod we join with D_Product
    df_abt = df_abt.withColumn("procedure_code", lit("PROC_CHECKUP")) \
                   .withColumn("diagnosis_code", lit("E11")) \
                   .withColumn("activity_type", lit("Service"))
    
    # Output
    output_path = dirs['data_output_dir'] / "abt_medical.parquet"
    df_abt.write.mode("overwrite").parquet(str(output_path))
    logging.info(f"Medical ABT saved to {output_path}")

if __name__ == "__main__":
    run_step()
