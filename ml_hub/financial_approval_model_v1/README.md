# Financial Approval Model (v1)

Predicts financial rejection of claims based on administrative rules and usage patterns.

## Steps
1. `steps/01_data_discovery.py`: Loads plan limits.
2. `steps/02_build_abt.py`: Simulates member/plan data.
3. `steps/03_feature_engineering.py`: Calculates utilization features.
4. `steps/train.py`: Trains LightGBM model.

## Usage
Run steps sequentially.
