# Chapter 2: The Core Hub (`medins_ml_utils`)

This chapter explores the heart of the system: the `medins_ml_utils` library. Every model depends on this code. We will examine it file by file.

## 2.1 `setup.py`: The Package Definition

We start with `ml_hub/medins_ml_utils/setup.py`. This file tells Python how to treat this directory as a library.

```python
from setuptools import setup, find_packages

setup(
    name="medins_ml_utils",
    version="0.2.0",
    packages=find_packages(),
    description="Shared utilities, data connectors, and feature logic for MedIns ML projects.",
    author="ML Platform Team",
    install_requires=[
        "pandas", "numpy", "scikit-learn", "lightgbm", "scipy", "mlflow", "pyspark"
    ],
    extras_require={
        "dev": ["pytest"],
    }
)
```

**Walkthrough:**
*   **`name="medins_ml_utils"`**: This is how we import it: `import medins_ml_utils`.
*   **`version="0.2.0"`**: Versioning is critical. If we change the logic for a feature, we bump the version. Models pin this version (e.g., `medins_ml_utils==0.2.0`) so they don't break unexpectedly.
*   **`packages=find_packages()`**: Automatically finds the inner `medins_ml_utils` directory.
*   **`install_requires`**: These are the dependencies the Hub itself needs. Note `pyspark` and `lightgbm` are core.

## 2.2 `utils.py`: The Boilerplate

Located at `ml_hub/medins_ml_utils/medins_ml_utils/utils.py`. This file handles configuration and file management.

```python
import json
from pathlib import Path
import logging
from pyspark.sql import SparkSession

def get_spark_session(app_name="MedInsMLHub"):
    """
    Creates or retrieves a SparkSession.
    Configures it for local simulation if no master is set.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
```
**Explanation:**
*   **SparkSession**: Every model needs Spark. Instead of every script writing `SparkSession.builder...`, we centralize it here.
*   **Arrow Optimization**: We enable Arrow (`spark.sql.execution.arrow.pyspark.enabled`) for faster conversion between Spark DataFrames and Pandas.

```python
def load_config(config_path: str = 'config.json') -> dict:
    """
    Loads the project configuration from a JSON file.
    Assumes the script is run from a project root where config.json is located.
    """
    try:
        full_config_path = Path.cwd() / config_path
        with open(full_config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {full_config_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the configuration file: {full_config_path}")
        raise
```
**Explanation:**
*   **`Path.cwd()`**: We use the current working directory. This assumes you run scripts from the model's root folder (e.g., `cd ml_hub/actuarial_pricing_model_v5`).
*   **Error Handling**: We catch `FileNotFoundError` and `JSONDecodeError` to give helpful error messages instead of crashing with a generic traceback.

```python
def setup_directories(config: dict) -> dict:
    """Creates all necessary output directories defined in the config."""
    paths = config['paths']
    project_root = Path.cwd()
    
    output_dir = project_root / paths['output_dir']
    data_output_dir = output_dir / "data"
    model_output_dir = output_dir / "models"
    results_output_dir = output_dir / "results"

    output_dir.mkdir(exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    results_output_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "data_dir": project_root / paths['data_dir'],
        "output_dir": output_dir,
        "data_output_dir": data_output_dir,
        "model_output_dir": model_output_dir,
        "results_output_dir": results_output_dir,
    }
```
**Explanation:**
*   **Standardization**: This enforces a standard folder structure for every model:
    *   `output/data`: Intermediate Parquet files.
    *   `output/models`: Saved `.txt` or `.pkl` models.
    *   `output/results`: Plots and metrics.
*   **`mkdir(parents=True, exist_ok=True)`**: Safely creates the folders if they don't exist.

```python
def save_artifacts(artifacts: dict, model_output_dir: Path):
    """Saves the calculated artifacts dictionary to a JSON file."""
    if not isinstance(model_output_dir, Path):
        model_output_dir = Path(model_output_dir)
        
    artifacts_path = model_output_dir / 'data_driven_artifacts.json'
    try:
        with open(artifacts_path, 'w') as f:
            json.dump(artifacts, f, indent=4)
        logging.info(f"Data-driven artifacts saved to {artifacts_path}")
    except Exception as e:
        logging.error(f"Failed to save artifacts: {e}")
        raise
```
**Explanation:**
*   **Artifacts**: These are global statistics learned in Step 1 (e.g., "The average cost of a heart attack"). We save them so Step 3 (Feature Engineering) can use them.
*   **Type Safety**: We check `isinstance(Path)` to be robust against sloppy callers passing strings.

## 2.3 `data_connectors.py`: The Gatekeeper

Located at `ml_hub/medins_ml_utils/medins_ml_utils/data_connectors.py`.

```python
import pandas as pd
from pathlib import Path
import logging
from pyspark.sql import SparkSession
from .utils import get_spark_session # Reuse our util

def get_data(data_dir: Path, tables_to_load: list, spark: SparkSession = None) -> dict:
    """
    The single, governed entry point for accessing data in the ML Hub.
    """
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    if spark is None:
        spark = get_spark_session()

    logging.info(f"SIMULATION: Loading {len(tables_to_load)} tables from central data directory: {data_dir.resolve()} using Spark...")
    try:
        data = {}
        for name in tables_to_load:
            # Handle potential variations in filenames (e.g., d_ vs D_)
            possible_filenames = [f"{name}.csv", f"d_{name}.csv", f"D_{name}.csv", f"f_{name}.csv", f"F_{name}.csv"]
            found = False
            for filename in possible_filenames:
                filepath = data_dir / filename
                if filepath.exists():
                    df = spark.read.option("header", "true") \
                                   .option("inferSchema", "true") \
                                   .csv(str(filepath))
                    data[name] = df
                    found = True
                    break
            if not found:
                 raise FileNotFoundError(f"Could not find a valid CSV for table '{name}' in {data_dir}")
        return data
    except FileNotFoundError as e:
        logging.error(f"A required data file was not found: {e}. Please check the 'ml_hub/data' directory.")
        raise
```
**Explanation:**
*   **The Abstraction**: The data scientist just asks for `get_data(..., ["Claims"])`. They don't care if it's a CSV, a Parquet, or a Snowflake table. This function handles the details.
*   **Robustness**: It tries multiple filename variations (`F_Claims.csv`, `claims.csv`) because data dumps are often inconsistent named.
*   **Spark Read**: `spark.read.csv` loads the data into a distributed DataFrame. We use `inferSchema=True` for convenience in this simulation, though in production, we would define a strict schema.

## 2.4 `features.py`: The Logic Core

Located at `ml_hub/medins_ml_utils/medins_ml_utils/features.py`. This is where the math lives.

We use **PySpark UDFs (User Defined Functions)**.

```python
# ... imports ...

def create_onboarding_features(df: DataFrame, artifacts: dict) -> DataFrame:
    """Creates features for new business applicants with limited data."""
    # ...
    cost_weights = artifacts['chronic_condition_cost_weights']
    freq_weights = artifacts['chronic_condition_freq_weights']
    
    @udf(FloatType())
    def calculate_self_reported_score(conditions):
        if not conditions:
            return 0.0
        score = 0.0
        for cond in conditions:
            cost_w = cost_weights.get(cond, 1.0)
            freq_w = freq_weights.get(cond, 1.0)
            score += (cost_w + freq_w) / 2.0
        return score

    df_out = df.withColumn('self_reported_chronic_score', calculate_self_reported_score(col('self_reported_conditions')))
    return df_out
```
**Explanation:**
*   **Closure**: The UDF `calculate_self_reported_score` *captures* the `cost_weights` dictionary from the outer scope. This is how we inject the global stats from Step 1 into the row-by-row logic.
*   **Logic**: It iterates through the patient's conditions. If they have Diabetes, look up the weight (e.g., 1.5). Add it to the score.
*   **Application**: `df.withColumn` applies this function to every row in the DataFrame.

**The Financial Features (New Additions):**

```python
def create_financial_approval_features(df: DataFrame, artifacts: dict) -> DataFrame:
    """Creates features for financial claim approval."""
    # ...
    plan_limits = artifacts.get('plan_limits', {})
    
    @udf(FloatType())
    def calculate_plan_utilization(plan_id, current_claim_amount, prior_spend):
        limit = plan_limits.get(plan_id, 10000.0)
        total_spend = (prior_spend if prior_spend else 0.0) + current_claim_amount
        return float(total_spend) / limit
    # ...
```
**Explanation:**
*   **Utilization**: `(prior_spend + current) / limit`. This tells us if the member is about to bust their cap.
*   **`artifacts` usage**: The `plan_limits` are loaded from the artifacts dictionary, which allows us to update plan rules without changing the code.

This concludes the tour of the Hub. It provides the **Tools** (Spark, Config) and the **Logic** (Features). Now let's see how a Spoke uses them.
