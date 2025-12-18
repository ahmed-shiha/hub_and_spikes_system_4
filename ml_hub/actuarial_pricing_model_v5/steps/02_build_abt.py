
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, collect_list, collect_set, sum as spark_sum, date_format, to_date, array_distinct, flatten, udf, coalesce, lit
from pyspark.sql.types import ArrayType, StringType

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
from medins_ml_utils.data_connectors import get_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = load_config()
    paths = setup_directories(config)
    spark = get_spark_session("PricingModel_Stage2_ABT")
    
    logging.info("--- Pricing Model: Stage 2 - Building Renewal & Onboarding ABTs (Spark) ---")

    tables_to_load = config['tables_to_load']['build_abt']
    data = get_data(Path(config['paths']['data_dir']), tables_to_load, spark)

    # Renaming for consistency with original pandas code
    # data['hidp_v2'] -> Patient_KEY
    hidp = data['hidp_v2'].withColumnRenamed('identitynumber', 'Patient_KEY')
    
    # F_Claim_Header -> Patient_KEY
    claim_header = data['F_Claim_Header']
    
    # Dynamic column mapping to find the ID column
    pk_col = None
    for c in claim_header.columns:
        if c.lower() == 'patient_key' or 'patient_key' in c.lower():
            pk_col = c
            break
    
    if pk_col and pk_col != 'Patient_KEY':
        claim_header = claim_header.withColumnRenamed(pk_col, 'Patient_KEY')

    # Dropping duplicates if they exist due to casing
    # If we have Patient_KEY and patient_key, rename failed or created duplicate if existing.
    # We will select exactly the columns we want to standardize.
    
    try:
        claim_header = claim_header.select(
            col('Patient_KEY'), 
            col('claim_request_bundle_id'), 
            col('payer_share_amount'), 
            col('max_serviced_date'), 
            col('provider_license_key')
        )
    except Exception as e:
        logging.warning(f"Selection failed: {e}. Attempting broad select.")
    
    careteam = data['F_CLAIM_CARETEAM'].withColumnRenamed('practitioner_license', 'PROVIDER_PRACTITIONER_LICENSE') \
                                       .withColumnRenamed('practice_desc', 'PRACTITIONER_SPECIALTY') \
                                       .withColumnRenamed('claim_ref', 'claim_request_bundle_id')
    
    claim_items = data['F_Claim_Item']
    
    # Join Header + Items
    claims = claim_header.join(claim_items, on='claim_request_bundle_id', how='left')
    
    # Join Careteam
    claims = claims.join(careteam, on='claim_request_bundle_id', how='left')
    
    # Clean Date
    claims = claims.withColumn('SERVICE_DATE', to_date(col('max_serviced_date')))
    
    claims_agg = claims.groupBy('Patient_KEY').agg(
        collect_set('principal_diagnosis_code').alias('diagnosis_codes_list'),
        collect_list('payer_share_amount').alias('claim_costs_list'),
        collect_list('SERVICE_DATE').alias('service_dates_list'),
        collect_set('PROVIDER_PRACTITIONER_LICENSE').alias('providers_visited_list'),
        collect_set('PRACTITIONER_SPECIALTY').alias('practitioner_specialty_list'),
    )
    
    # Missing columns handling
    for c in ['pre_auth_outcomes_list', 'activity_type_list', 'adjudication_reason_list']:
        if c not in claims_agg.columns:
            claims_agg = claims_agg.withColumn(c, lit([]).cast(ArrayType(StringType())))
    
    # Patient Aggregation (Cost)
    costs_agg = claim_header.groupBy('Patient_KEY').agg(spark_sum('payer_share_amount').alias('annualized_cost'))
    
    # Patient Base
    # Check if `hidp` has another patient key column?
    if 'Patient_KEY' in data['hidp_v2'].columns:
        hidp = data['hidp_v2'] # Use as is
        if 'identitynumber' in hidp.columns:
            hidp = hidp.drop('identitynumber')
    else:
        hidp = data['hidp_v2'].withColumnRenamed('identitynumber', 'Patient_KEY')
        
    patient_df = hidp.join(costs_agg, on='Patient_KEY', how='left').na.fill({'annualized_cost': 0})
    
    abt_renewal = patient_df.join(claims_agg, on='Patient_KEY', how='left')
    
    # Handle Null Lists
    list_cols = [c for c in abt_renewal.columns if c.endswith('_list')]
    for c in list_cols:
        abt_renewal = abt_renewal.withColumn(c, coalesce(col(c), lit([]).cast(abt_renewal.schema[c].dataType)))

    # Save Renewal ABT
    abt_renewal_path = str(Path(paths['data_output_dir']) / 'abt_renewal.parquet')
    abt_renewal.write.mode("overwrite").parquet(abt_renewal_path)
    logging.info(f"Renewal ABT saved to {abt_renewal_path}")

    # Onboarding ABT
    chronic_defs = config['business_logic']['chronic_conditions']
    
    @udf(ArrayType(StringType()))
    def derive_conditions(diags):
        if not diags: return []
        found = set()
        for cond, codes in chronic_defs.items():
            if any(str(d).startswith(tuple(codes)) for d in diags if d):
                found.add(cond)
        return list(found)

    abt_renewal = abt_renewal.withColumn('self_reported_conditions', derive_conditions(col('diagnosis_codes_list')))
    
    abt_onboarding = abt_renewal.select('Patient_KEY', 'age', 'gender', 'city', 'classname', 'self_reported_conditions', 'annualized_cost')
    
    abt_onboarding_path = str(Path(paths['data_output_dir']) / 'abt_onboarding.parquet')
    abt_onboarding.write.mode("overwrite").parquet(abt_onboarding_path)
    logging.info(f"Onboarding ABT saved to {abt_onboarding_path}")

    logging.info("--- Stage 2 Complete ---")

if __name__ == "__main__":
    main()
