
import pandas as pd
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)

PREDICTION_LOG_FILE = "fwa_prediction_log.jsonl"

def main():
    logging.info("--- FWA Model Monitoring: Checking for Data Drift ---")
    
    # Simulate loading baseline training data distribution
    training_data_avg_cost_mean = 1500.0
    
    log_path = Path(PREDICTION_LOG_FILE)
    if not log_path.exists():
        logging.warning("Prediction log file not found. Cannot check for drift.")
        return

    try:
        with open(log_path, 'r') as f:
            logs = [json.loads(line) for line in f]
    except (json.JSONDecodeError, IOError):
        logs = []
    
    if not logs:
        logging.info("No requests logged. Cannot check for drift.")
        return
        
    live_costs = [log['request_body']['avg_cost_per_claim'] for log in logs]
    if not live_costs:
        return
        
    live_mean_cost = pd.Series(live_costs).mean()
    
    logging.info(f"Baseline training data mean avg_cost_per_claim: {training_data_avg_cost_mean:.2f}")
    logging.info(f"Live data mean avg_cost_per_claim: {live_mean_cost:.2f}")

    drift = abs(live_mean_cost - training_data_avg_cost_mean) / training_data_avg_cost_mean
    
    if drift > 0.15: # 15% drift threshold
        logging.warning(f"CRITICAL ALERT: Data drift detected! Mean avg_cost_per_claim has drifted by {drift:.2%}, exceeding threshold.")
    else:
        logging.info("No significant data drift detected for FWA model.")

if __name__ == "__main__":
    main()
