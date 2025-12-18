
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, lit, collect_list, sum as spark_sum, countDistinct, when, array_contains, avg, size, udf
from pyspark.sql.types import BooleanType

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, save_artifacts, get_spark_session
from medins_ml_utils.data_connectors import get_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_artifacts(train_df, config: dict) -> dict:
    logging.info("Calculating data-driven artifacts from training data (Spark)...")
    chronic_conditions = config['business_logic']['chronic_conditions']
    
    overall_avg_cost_row = train_df.select(avg(col('annualized_cost'))).collect()[0]
    overall_avg_cost = overall_avg_cost_row[0] if overall_avg_cost_row[0] else 0.0
    
    chronic_cost_weights = {}
    
    for condition, codes in chronic_conditions.items():
        condition_codes = codes 
        
        @udf(BooleanType())
        def has_condition(diags):
            if not diags: return False
            return any(str(d).startswith(tuple(condition_codes)) for d in diags)
            
        avg_cost_with_cond_row = train_df.filter(has_condition(col('diagnosis_codes_list'))) \
                                         .select(avg(col('annualized_cost'))).collect()[0]
        
        avg_cost_with_condition = avg_cost_with_cond_row[0] if avg_cost_with_cond_row[0] else 0.0
        
        weight = max(1.0, avg_cost_with_condition / overall_avg_cost if overall_avg_cost > 0 else 1.0)
        chronic_cost_weights[condition] = round(float(weight), 2)
        
    logging.info(f"Calculated chronic cost weights: {chronic_cost_weights}")

    chronic_freq_weights = {k: v for k, v in chronic_cost_weights.items()}

    return {
        "chronic_conditions_definitions": chronic_conditions,
        "chronic_condition_cost_weights": chronic_cost_weights,
        "chronic_condition_freq_weights": chronic_freq_weights,
        "denial_codes": config['business_logic'].get('denial_codes', {})
    }

def main():
    config = load_config()
    paths = setup_directories(config)
    spark = get_spark_session("PricingModel_Stage1")
    
    logging.info("--- Pricing Model: Stage 1 - Data Discovery & Artifact Creation (Spark) ---")
    
    tables_to_load = config['tables_to_load']['data_discovery']
    raw_data = get_data(Path(config['paths']['data_dir']), tables_to_load, spark)

    patients = raw_data['D_Patient'].select("Patient_KEY").distinct()
    train_patients_df, test_patients_df = patients.randomSplit([0.7, 0.3], seed=42)
    
    train_patients_df.toPandas().to_csv(Path(paths['data_output_dir']) / 'train_member_ids.csv', index=False)
    test_patients_df.toPandas().to_csv(Path(paths['data_output_dir']) / 'test_member_ids.csv', index=False)
    
    logging.info(f"Split members.")

    # Fix: claims table 'Patient_KEY' handling
    claims = raw_data['F_Claim_Header']
    
    # Try to find the patient id col
    pk_col = None
    for c in claims.columns:
        if c.lower() == 'patient_key' or 'patient_key' in c.lower():
            pk_col = c
            break
            
    if pk_col and pk_col != 'Patient_KEY':
        claims = claims.withColumnRenamed(pk_col, 'Patient_KEY')
    elif pk_col is None:
        logging.error("No patient key found in claims.")
        return

    # Handle potential duplicates if renaming didn't work as expected or column existed
    if 'Patient_KEY' in claims.columns:
         pass 
    elif 'Patient_KEY0' in claims.columns: # Spark renamed it?
         claims = claims.withColumnRenamed('Patient_KEY0', 'Patient_KEY')
         
    claims_train = claims.join(train_patients_df, on="Patient_KEY", how="inner")
    
    claim_items = raw_data['F_Claim_Item']
    diagnosis = raw_data['D_Diagnosis_Code']

    # Rename columns to standard for join
    claim_items = claim_items.withColumnRenamed("Diagnosis_Code", "principal_diagnosis_code")
    diagnosis = diagnosis.withColumnRenamed("Diagnosis_Code", "code")
    
    claim_diag = claim_items.join(diagnosis, claim_items.principal_diagnosis_code == diagnosis.code, "left")
    
    # Check join keys
    join_key_bundle = 'claim_request_bundle_id' if 'claim_request_bundle_id' in claim_diag.columns else 'Claim_ID'
    
    diag_agg = claim_diag.groupBy(join_key_bundle).agg(collect_list("code").alias("diagnosis_codes_list"))
    
    # Align header column for join
    if 'Claim_ID' in diag_agg.columns:
         diag_agg = diag_agg.withColumnRenamed('Claim_ID', 'claim_request_bundle_id')
    
    claims_train_diag = claims_train.join(diag_agg, on="claim_request_bundle_id", how="left")
    
    patient_stats = claims_train_diag.groupBy("Patient_KEY").agg(
        spark_sum("payer_share_amount").alias("annualized_cost"),
        countDistinct("claim_request_bundle_id").alias("claim_count")
    )
    
    claims_for_artifacts = claims_train_diag.join(patient_stats, on="Patient_KEY", how="left")
    claims_for_artifacts.cache()

    artifacts = calculate_artifacts(claims_for_artifacts, config)
    
    # --- Enhancement: Calculate Provider Specialty Baselines (The Fingerprint) ---
    # We use the General Data to learn: "How much does a Cardiologist typically cost?"
    # This helps us price new providers who have no history but have a specialty.
    
    if 'd_provider' in raw_data:
        providers = raw_data['d_provider']
        # Join claims with providers to get specialty
        # Assuming F_Claim_Header has 'provider_license_key' and d_provider has 'Provider_ID'
        
        # Standardize join keys
        prov_key_col = 'Provider_ID'
        if 'provider_id' in providers.columns: prov_key_col = 'provider_id'
        
        # Rename for join
        providers = providers.withColumnRenamed(prov_key_col, 'provider_license_key')
        
        claims_prov = claims.join(providers, on='provider_license_key', how='inner')
        
        if 'Specialty' in claims_prov.columns:
            specialty_stats = claims_prov.groupBy('Specialty').agg(
                avg('payer_share_amount').alias('avg_cost')
            ).collect()
            
            specialty_avg_costs = {row['Specialty']: row['avg_cost'] for row in specialty_stats}
            artifacts['specialty_avg_costs'] = specialty_avg_costs
            logging.info(f"Calculated specialty fingerprints: {specialty_avg_costs}")
        else:
            logging.warning("Specialty column not found in provider data. Skipping specialty fingerprints.")
            artifacts['specialty_avg_costs'] = {}
    else:
        logging.warning("d_provider table not found. Skipping specialty fingerprints.")
        artifacts['specialty_avg_costs'] = {}

    save_artifacts(artifacts, Path(paths['model_output_dir']))
    
    logging.info("--- Stage 1 Complete ---")

if __name__ == "__main__":
    main()
