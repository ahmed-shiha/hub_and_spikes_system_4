
import pandas as pd
import logging
import sys
from pathlib import Path
from pyspark.sql.functions import col, count, when, isnan

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
from medins_ml_utils.data_connectors import get_data

logging.basicConfig(level=logging.INFO)

def main():
    config = load_config()
    paths = setup_directories(config)
    spark = get_spark_session("Medical_EDA")
    
    logging.info("--- Medical Model: Stage 0 - Exploratory Data Analysis ---")
    
    # Load Raw Data
    # We look at the new raw_data folder if available
    data_dir = Path(config['paths']['data_dir'])
    raw_dir = data_dir / "raw_data"
    
    # Check if raw_data exists, else fallback
    if raw_dir.exists():
        logging.info(f"Found Raw Data at {raw_dir}")
        # Analyze d_product_or_service
        try:
            services = spark.read.csv(str(raw_dir / "d_product_or_service.csv"), header=True, inferSchema=True)
            
            # Profile: Activity Types
            logging.info("Profile: Service Activity Types")
            services.groupBy("activity_type").count().show()
            
            # Check for nulls
            total = services.count()
            nulls = services.filter(col("activity_type").isNull()).count()
            logging.info(f"Null Activity Types: {nulls} / {total}")
            
        except Exception as e:
            logging.warning(f"Could not load raw service file: {e}")
            
        # Analyze Pre-Auth (Medical History)
        try:
            pre_auth = spark.read.csv(str(raw_dir / "f_pre_auth_header.csv"), header=True, inferSchema=True)
            logging.info("Profile: Pre-Auth Adjudication Outcomes")
            pre_auth.groupBy("adjudication_outcome").count().show()
        except Exception as e:
            logging.warning(f"Could not load pre-auth file: {e}")

    else:
        logging.warning("Raw Data folder not found. Using simulated data.")
        
    logging.info("EDA Complete. Report would be saved to output/results/eda_report.html")

if __name__ == "__main__":
    main()
