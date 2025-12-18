
import sys
from pathlib import Path
import logging
import pandas as pd
import ast
from pyspark.sql.functions import col, udf, lit
from pyspark.sql.types import ArrayType, StringType

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, load_artifacts, get_spark_session
from medins_ml_utils.features import (
    create_onboarding_features, 
    create_renewal_vice_champion_features,
    create_renewal_champion_features
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = load_config()
    paths = setup_directories(config)
    artifacts = load_artifacts(Path(paths['model_output_dir']))
    spark = get_spark_session("PricingModel_Stage3_Features")
    
    logging.info("--- Pricing Model: Stage 3 - Feature Engineering Orchestration (Spark) ---")

    # Read Parquet inputs (output from Step 2)
    abt_renewal_path = str(Path(paths['data_output_dir']) / 'abt_renewal.parquet')
    abt_onboarding_path = str(Path(paths['data_output_dir']) / 'abt_onboarding.parquet')
    
    abt_renewal = spark.read.parquet(abt_renewal_path)
    abt_onboarding = spark.read.parquet(abt_onboarding_path)

    # Note: Parquet preserves schema (Arrays are Arrays), so no need for ast.literal_eval.
    
    logging.info("Generating Onboarding Champion features...")
    # medins_ml_utils.features functions now accept Spark DataFrame
    abt_onboarding_featured = create_onboarding_features(abt_onboarding, artifacts)
    
    # Save Feature Set
    # We save as CSV for final training step (Step 5) compatibility if it expects CSV and fits in memory.
    # But for "millions of rows", Parquet is better. Step 5 should ideally read Parquet.
    # I will stick to CSV for the *featured* output if I want to support easy inspection, 
    # but Parquet is the correct technical choice. I will use Parquet.
    # But to satisfy the user request of "tests, edge cases" and "detailed plan", 
    # I should ensure downstream consumes it.
    
    abt_onboarding_featured.write.mode("overwrite").parquet(str(Path(paths['data_output_dir']) / 'abt_onboarding_featured.parquet'))

    logging.info("Generating Renewal Vice-Champion features (no provider)...")
    abt_renewal_vice_featured = create_renewal_vice_champion_features(abt_renewal, artifacts)
    abt_renewal_vice_featured.write.mode("overwrite").parquet(str(Path(paths['data_output_dir']) / 'abt_renewal_vice_champion_featured.parquet'))
    
    logging.info("Generating Renewal Champion features (with provider)...")
    # Note: Renewal Champion builds on Vice Champion + Original ABT (for provider lists)
    # In Spark, we just pass the DF which has all cols.
    abt_renewal_champ_featured = create_renewal_champion_features(abt_renewal_vice_featured, abt_renewal, artifacts)
    abt_renewal_champ_featured.write.mode("overwrite").parquet(str(Path(paths['data_output_dir']) / 'abt_renewal_champion_featured.parquet'))

    logging.info("--- Stage 3 Complete ---")

if __name__ == "__main__":
    main()
