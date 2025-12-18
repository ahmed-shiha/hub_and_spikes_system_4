
import logging
from medins_ml_utils.utils import load_config

logging.basicConfig(level=logging.INFO)

def main():
    config = load_config()
    logging.info("--- Pricing Model Validation: Comparing Candidate vs. Production Champion ---")
    try:
        # SIMULATE loading model performance from MLflow
        prod_mae = 1000.0
        candidate_mae = 950.0
        
        logging.info(f"Production Champion MAE: {prod_mae:.2f}")
        logging.info(f"New Candidate MAE: {candidate_mae:.2f}")

        improvement = (prod_mae - candidate_mae) / prod_mae
        threshold = config['validation_thresholds']['performance_improvement_pct']

        if improvement < threshold:
            logging.error(f"Validation FAILED: Performance change {improvement:.2%} does not meet threshold of {threshold:.2%}.")
            exit(1)
        else:
            logging.info(f"Validation PASSED: Performance change {improvement:.2%} meets threshold.")
    except Exception as e:
        logging.error(f"Failed to validate models: {e}")
        exit(1)

if __name__ == "__main__":
    main()
