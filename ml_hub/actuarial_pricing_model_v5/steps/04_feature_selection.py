
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = load_config()
    paths = setup_directories(config)
    spark = get_spark_session("PricingModel_Stage4_FeatureSelection")
    
    logging.info("--- Pricing Model: Stage 4 - Feature Selection & Analysis ---")
    
    # Load Featured Data (Parquet)
    # We analyze 'renewal_champion' as it has the most features.
    input_path = str(Path(paths['data_output_dir']) / 'abt_renewal_champion_featured.parquet')
    if not Path(input_path).exists():
        logging.warning("Featured data not found. Skipping Stage 4.")
        return

    df = spark.read.parquet(input_path)
    
    # 1. Feature Correlation Analysis
    # For large data, we sample or compute correlation matrix in Spark.
    # Spark MLlib Correlation.
    from pyspark.ml.stat import Correlation
    from pyspark.ml.feature import VectorAssembler
    
    # Select numeric columns
    numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ['int', 'double', 'float', 'bigint', 'long']]
    # Exclude ID and Target for feature-feature correlation, but include Target for Feature-Target.
    numeric_cols = [c for c in numeric_cols if c not in ['Patient_KEY']]
    
    if not numeric_cols:
        logging.warning("No numeric columns found for correlation analysis.")
        return

    # Assemble vector
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid="skip")
    df_vector = assembler.transform(df).select("features")
    
    # Calculate Correlation Matrix
    # This can be expensive on millions of rows but feasible.
    logging.info(f"Calculating correlation matrix for {len(numeric_cols)} features...")
    matrix = Correlation.corr(df_vector, "features").head()
    
    # Extract matrix to local for visualization/saving
    corr_matrix = matrix[0].toArray()
    corr_df = pd.DataFrame(corr_matrix, index=numeric_cols, columns=numeric_cols)
    
    # Save Correlation Matrix
    corr_df.to_csv(Path(paths['results_output_dir']) / 'feature_correlations.csv')
    
    # 2. Feature Importance (Simple Filter Method)
    # Correlation with Target 'annualized_cost'
    if 'annualized_cost' in numeric_cols:
        target_corr = corr_df['annualized_cost'].sort_values(ascending=False)
        target_corr.to_csv(Path(paths['results_output_dir']) / 'feature_target_correlation.csv')
        logging.info("Top 5 correlates with cost:")
        logging.info(target_corr.head(6))
        
    logging.info("--- Stage 4 Complete ---")

if __name__ == "__main__":
    main()
