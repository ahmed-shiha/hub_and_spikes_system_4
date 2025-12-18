
import pandas as pd
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)

PREDICTION_LOG_FILE = "prediction_log.jsonl"

def main():
    logging.info("--- Pricing Model Monitoring: Checking for Data Drift ---")
    
    # Simulate loading baseline training data distribution
    training_data_age_mean = 35.0
    
    # Simulate reading live data from observability logs
    log_path = Path(PREDICTION_LOG_FILE)
    if not log_path.exists():
        logging.warning("Prediction log file not found. Cannot check for drift.")
        return

    with open(log_path, 'r') as f:
        logs = [json.loads(line) for line in f]
    
    live_ages = [log['request_body']['age'] for log in logs]
    if not live_ages:
        logging.info("No requests logged. Cannot check for drift.")
        return
        
    live_mean_age = pd.Series(live_ages).mean()
    
    logging.info(f"Baseline training data mean age: {training_data_age_mean:.2f}")
    logging.info(f"Live data mean age: {live_mean_age:.2f}")

    drift = abs(live_mean_age - training_data_age_mean) / training_data_age_mean
    
    if drift > 0.10: # 10% drift threshold
        logging.warning(f"CRITICAL ALERT: Data drift detected! Mean age has drifted by {drift:.2%}, exceeding threshold.")
    else:
        logging.info("No significant data drift detected.")

if __name__ == "__main__":
    main()
