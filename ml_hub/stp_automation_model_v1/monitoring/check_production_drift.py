
import pandas as pd
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)

PREDICTION_LOG_FILE = "stp_prediction_log.jsonl"

def main():
    logging.info("--- STP Model Monitoring: Checking for Data Drift ---")
    
    # Simulate loading baseline training data distribution
    training_data_amount_mean = 850.0
    
    log_path = Path(PREDICTION_LOG_FILE)
    if not log_path.exists():
        logging.warning("Prediction log file not found. Cannot check for drift.")
        return

    try:
        with open(log_path, 'r') as f:
            # In a real system, you would parse the 'request_body' from the log
            # For simulation, we'll assume a simplified log structure for clarity
            logs = [json.loads(line) for line in f] 
    except (json.JSONDecodeError, IOError):
        logs = []
    
    if not logs:
        logging.info("No requests logged. Cannot check for drift.")
        return
        
    live_amounts = [log.get('submitted_amount', 0) for log in logs] # Simplified for demo
    if not live_amounts:
        return
        
    live_mean_amount = pd.Series(live_amounts).mean()
    
    logging.info(f"Baseline training data mean submitted_amount: {training_data_amount_mean:.2f}")
    logging.info(f"Live data mean submitted_amount: {live_mean_amount:.2f}")

    drift = abs(live_mean_amount - training_data_amount_mean) / training_data_amount_mean
    
    if drift > 0.20: # 20% drift threshold
        logging.warning(f"CRITICAL ALERT: Data drift detected! Mean submitted_amount has drifted by {drift:.2%}, exceeding threshold.")
    else:
        logging.info("No significant data drift detected for STP model.")

if __name__ == "__main__":
    main()
