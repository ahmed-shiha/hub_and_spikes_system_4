
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
from medins_ml_utils.data_connectors import get_data
from pyspark.sql.functions import col, rand, lit, array
import logging

def run_step():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    dirs = setup_directories(config)
    spark = get_spark_session("Pricing_ABT_Onboarding")
    
    # Load Data? For onboarding, we typically assume we get data from an Application Form (CSV)
    # Since we don't have an application table, we will simulate it from the Member table (Patient_KEY)
    # assuming these are "new" members.
    
    data = get_data(dirs['data_dir'], ["F_Claim_Header"], spark)
    # We just need distinct patients to simulate applicants
    df_claim_header = data["F_Claim_Header"]
    
    # Determine the column name (handle Spark case sensitivity/dup issue)
    target_col = 'Patient_KEY0' if 'Patient_KEY0' in df_claim_header.columns else 'Patient_KEY'
    
    df_applicants = df_claim_header.select(col(target_col).alias("Patient_KEY")).distinct()
    
    # Simulate Application Data
    # Age, Gender, Region, Self-Reported Conditions
    df_abt = df_applicants.withColumn("age", (rand() * 60 + 20).cast("int")) \
                          .withColumn("gender", (rand() * 2).cast("int")) \
                          .withColumn("region", lit("R1")) \
                          .withColumn("self_reported_conditions", array(lit("E11"), lit("I10"))) # Simulate everyone admits to something for testing
    
    # Target: We pretend we know their future cost (from the claims table) for training purposes
    # In reality, we join with future claims. Here we simulate 'future_annual_cost'.
    df_abt = df_abt.withColumn("future_annual_cost", (rand() * 5000 + 1000).cast("float"))

    # Output
    output_path = dirs['data_output_dir'] / "abt_pricing_onboarding.parquet"
    df_abt.write.mode("overwrite").parquet(str(output_path))
    logging.info(f"Onboarding ABT saved to {output_path}")

if __name__ == "__main__":
    run_step()
