
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
from medins_ml_utils.data_connectors import get_data
from pyspark.sql.functions import col, rand
import logging

def run_step():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    dirs = setup_directories(config)
    spark = get_spark_session("FinancialApproval_ABT")
    
    # Load Data
    data = get_data(dirs['data_dir'], ["F_Claim_Header"], spark)
    df = data["F_Claim_Header"]
    
    # Simulate joining Member/Plan data
    # We add columns needed for financial features
    # Add dummy columns for new features
    from pyspark.sql.functions import lit, array, md5, concat_ws
    
    df_abt = df.withColumn("plan_id", (rand() * 3).cast("int").cast("string")) \
               .withColumn("ytd_spend", (rand() * 5000).cast("float")) \
               .withColumn("claim_amount", col("payer_share_amount").cast("float")) \
               .withColumn("claim_hash", md5(concat_ws("|", col("max_serviced_date"), col("payer_share_amount"), col("provider_license_key")))) \
               .withColumn("history_hashes", array(lit("dummy_hash")))
               
    # Output
    output_path = dirs['data_output_dir'] / "abt_financial.parquet"
    df_abt.write.mode("overwrite").parquet(str(output_path))
    logging.info(f"ABT saved to {output_path}")

if __name__ == "__main__":
    run_step()
