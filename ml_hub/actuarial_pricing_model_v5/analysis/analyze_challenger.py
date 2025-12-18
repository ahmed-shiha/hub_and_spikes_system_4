
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("--- Pricing Model Analysis: Comparing Champion vs. Challenger Live Performance ---")
    
    # In production, this would join the prediction logs with actual outcomes from a data warehouse.
    # For simulation, we create a mock DataFrame.
    live_results = pd.DataFrame({
        "champion_prediction": [1000, 2000, 1500, 3000],
        "challenger_prediction": [950, 2050, 1520, 2900],
        "actual_cost": [980, 2100, 1500, 2950]
    })

    champion_mae = mean_absolute_error(live_results['actual_cost'], live_results['champion_prediction'])
    challenger_mae = mean_absolute_error(live_results['actual_cost'], live_results['challenger_prediction'])
    
    print("\n--- Live Performance Results ---")
    print(f"Champion MAE:   {champion_mae:,.2f} SAR")
    print(f"Challenger MAE: {challenger_mae:,.2f} SAR")
    print("-" * 30)

    if challenger_mae < champion_mae:
        improvement = (champion_mae - challenger_mae) / champion_mae
        print(f"Recommendation: ✅ Promote Challenger. It is outperforming the Champion by {improvement:.2%}.")
    else:
        print("Recommendation: ❌ Do Not Promote. The Champion continues to perform better or equally.")

if __name__ == "__main__":
    main()
