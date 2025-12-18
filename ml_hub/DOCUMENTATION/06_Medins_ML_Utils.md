# Reference: MedIns ML Utils (The Hub)

## 1. Overview
`medins_ml_utils` is the "Hub" in the MedIns "Hub & Spoke" architecture. It is a shared Python library that centralizes:
*   **Data Access:** Governed connectors to data warehouses (simulated via local CSVs).
*   **Feature Engineering:** Standardized PySpark UDFs to ensure features are calculated consistently across models (e.g., Pricing vs. Churn).
*   **Online Features:** Connectors to low-latency stores (e.g., Redis) for real-time inference.
*   **Utilities:** Common boilerplate for configuration, logging, and artifact management.

This library ensures that a feature like "Chronic Condition Score" is defined in *code* exactly once, preventing logic drift between different teams.

## 2. Installation
The library is designed to be installed in "editable" mode (`-e`) during development so that changes in the Hub are immediately reflected in the Spokes.

```bash
# From the ml_hub root directory
pip install -e medins_ml_utils
```

## 3. Module Reference

### 3.1 `data_connectors`
Handles governed access to raw data.

*   `get_spark_session(app_name="MedInsMLHub")`
    *   Creates or retrieves a standard PySpark session configured for the environment.
*   `get_data(data_dir: Path, tables_to_load: list, spark: SparkSession = None) -> dict`
    *   **Production Logic:** Connects to Snowflake/BigQuery.
    *   **Simulation Logic:** Reads CSV files from the central `ml_hub/data/` directory.
    *   **Returns:** A dictionary mapping table names to Spark DataFrames.

### 3.2 `features`
Contains the core business logic for transforming raw data into mathematical features. These are implemented using PySpark functions and UDFs for scalability.

*   `create_onboarding_features(df: DataFrame, artifacts: dict) -> DataFrame`
    *   Calculates risk scores for new members based on self-reported conditions.
    *   **Key Feature:** `self_reported_chronic_score`.
*   `create_renewal_vice_champion_features(df: DataFrame, artifacts: dict) -> DataFrame`
    *   Calculates advanced history-based features (Vice-Champion model).
    *   **Key Features:** `data_driven_risk_score`, `cost_volatility_iqr`, `days_between_visits_std`, `cost_trend_slope`, `care_team_complexity`.
*   `create_renewal_champion_features(df: DataFrame, original_abt: DataFrame, artifacts: dict) -> DataFrame`
    *   Adds provider-specific risk features (Champion model).
    *   **Key Features:** `avg_provider_risk_score` (Weighted by provider credibility), `hospital_visit_ratio`.
*   `safe_literal_eval(val)`
    *   Safely parses string representations of lists (e.g., `"['A', 'B']"`) into actual Python lists.

### 3.3 `online_features`
Simulates access to a real-time feature store (like Redis) for low-latency inference.

*   `get_member_claims_in_last_hour(member_id: str) -> int`
    *   **Production Logic:** Query Redis key `claims_last_hour:{member_id}`.
    *   **Simulation Logic:** Returns a random integer.
    *   **Use Case:** Used in fraud detection or STP automation where recent activity spikes indicate risk.

### 3.4 `utils`
General-purpose utilities to standardize project structure.

*   `load_config(config_path: str = 'config.json') -> dict`
    *   Loads JSON configuration.
*   `setup_directories(config: dict) -> dict`
    *   Creates standard output folders: `data/`, `models/`, `results/`.
*   `save_artifacts(artifacts: dict, model_output_dir: Path)`
    *   Saves calculated statistics (e.g., disease weights) to JSON.
*   `load_artifacts(model_output_dir: Path) -> dict`
    *   Loads previously saved artifacts.

## 4. Usage Example
How a "Spoke" model (e.g., Actuarial Pricing) uses the Hub:

```python
from medins_ml_utils import data_connectors, features, utils

# 1. Setup
spark = data_connectors.get_spark_session("PricingModel")
config = utils.load_config()

# 2. Get Data
data_map = data_connectors.get_data(
    data_dir=Path(config['paths']['data_dir']),
    tables_to_load=["F_Claim_Header", "F_Claim_Item"],
    spark=spark
)

# 3. Apply Feature Engineering (Standardized)
artifacts = utils.load_artifacts(Path(config['paths']['output_dir']) / "models")
featured_df = features.create_renewal_vice_champion_features(
    df=data_map["F_Claim_Header"], 
    artifacts=artifacts
)
```
