
import pandas as pd
from sklearn.metrics import mean_absolute_error
import logging
from medins_ml_utils.utils import load_config

logging.basicConfig(level=logging.INFO)

def main():
    config = load_config()
    logging.info("--- Pricing Model Validation: Checking for Model Bias ---")
    
    # Simulate loading evaluation data
    eval_df = pd.DataFrame({
        'y_true': [1000, 2000, 1500, 2500, 1200, 2200],
        'y_pred': [1100, 1900, 1600, 2700, 1300, 2100],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female']
    })
    
    overall_mae = mean_absolute_error(eval_df['y_true'], eval_df['y_pred'])
    gender_mae = eval_df.groupby('gender').apply(lambda g: mean_absolute_error(g['y_true'], g['y_pred']))
    
    max_disparity = 0
    for group, mae in gender_mae.items():
        disparity = abs(mae - overall_mae) / overall_mae
        if disparity > max_disparity:
            max_disparity = disparity
        logging.info(f"MAE for {group}: {mae:.2f} (Disparity: {disparity:.2%})")

    bias_threshold = config['validation_thresholds']['bias_disparity_max']
    if max_disparity > bias_threshold:
        logging.error(f"Validation FAILED: Max bias disparity {max_disparity:.2%} exceeds threshold of {bias_threshold:.2%}.")
        exit(1)
    else:
        logging.info("Validation PASSED: Model bias is within acceptable limits.")

if __name__ == "__main__":
    main()
