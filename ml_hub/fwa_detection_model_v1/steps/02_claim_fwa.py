
import pandas as pd
from sklearn.ensemble import IsolationForest
import logging
from pathlib import Path
import sys
from pyspark.sql.functions import col, lit

# Add project root for robust imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config, setup_directories, get_spark_session
from medins_ml_utils.data_connectors import get_data
from medins_ml_utils.features import create_claim_fwa_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    config = load_config('config.json')
    paths = setup_directories(config)
    spark = get_spark_session("FWA_Stage2_Claim")
    
    logging.info("--- FWA Stage 2: Claim Level Detection ---")
    
    # Load data
    data_dir = Path(config['paths']['data_dir'])
    raw_data = get_data(data_dir, ['F_Claim_Header'], spark)
    df = raw_data['F_Claim_Header']
    
    # Feature Engineering
    df_features = create_claim_fwa_features(df, {})
    
    # Unsupervised Detection
    logging.info("Running Isolation Forest on Claims...")
    pdf = df_features.select('is_round_amount', 'starts_with_nine', 'payer_share_amount', 'claim_request_bundle_id').toPandas()
    
    X = pdf[['is_round_amount', 'starts_with_nine', 'payer_share_amount']].fillna(0)
    
    model = IsolationForest(contamination=0.001) # Very rare
    model.fit(X)
    
    pdf['anomaly_score'] = model.decision_function(X)
    suspicious_claims = pdf.sort_values('anomaly_score').head(50)
    
    output_path = Path(paths['output_dir']) / "suspicious_claims.csv"
    suspicious_claims.to_csv(output_path, index=False)
    logging.info(f"Saved {len(suspicious_claims)} suspicious claims to {output_path}")

if __name__ == "__main__":
    main()
