# Chapter 3: The First Gate - Financial Approval

The **Financial Approval Model** is the first line of defense. Before we even ask "Is this medical treatment necessary?", we ask "Is this claim administratively valid?".

Location: `ml_hub/financial_approval_model_v1`.

## 3.1 `config.json`

This file controls the pipeline.

```json
{
    "project_name": "financial_approval_model_v1",
    "paths": {
        "data_dir": "../data",
        "output_dir": "output"
    },
    "model_params": {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05
    },
    "validation_thresholds": {
        "min_auc": 0.80
    }
}
```
**Explanation:**
*   **`data_dir`**: Points to `../data` because the model folder is nested one level deep inside `ml_hub`.
*   **`model_params`**: These are passed directly to LightGBM. `binary` means we predict Yes/No (Reject/Approve).
*   **`validation_thresholds`**: If the AUC is below 0.80, the pipeline will raise an error and refuse to deploy. This is a quality gate.

## 3.2 Step 1: Data Discovery (`steps/01_data_discovery.py`)

This step learns the "Rules of the Game".

```python
from medins_ml_utils.utils import load_config, setup_directories, save_artifacts, get_spark_session
# ...

def run_step():
    # ... setup code ...
    
    # Discovery: Calculate Plan Limits (Simulated)
    # in reality, this comes from a Plan Master table.
    artifacts = {
        "plan_limits": {
            "P001": 5000.0,
            "P002": 10000.0,
            "P003": 1000000.0
        },
        "provider_master": {
            "PROV1": {"tier": 1},
            "PROV2": {"tier": 2},
            "PROV3": {"tier": 1},
            "PROV4": {"tier": 3} # Blacklisted
        }
    }
    
    save_artifacts(artifacts, config['paths']['output_dir'] + "/models")
```
**Explanation:**
*   **Goal**: Create the `data_driven_artifacts.json`.
*   **Simulated Logic**: In a real system, we would query a SQL database `SELECT * FROM PLAN_MASTER`. Here, we hardcode a dictionary for simulation.
*   **Provider Tiers**: We define that `PROV1` is Tier 1 (Preferred) and `PROV4` is Tier 3 (Blacklisted). This lookup table is saved to be used in feature engineering.

## 3.3 Step 2: Build ABT (`steps/02_build_abt.py`)

This step creates the training dataset.

```python
# ... imports ...
def run_step():
    # ... setup ...
    data = get_data(dirs['data_dir'], ["F_Claim_Header"], spark)
    df = data["F_Claim_Header"]
    
    from pyspark.sql.functions import lit, array, md5, concat_ws, rand
    
    df_abt = df.withColumn("plan_id", (rand() * 3).cast("int").cast("string")) \
               .withColumn("ytd_spend", (rand() * 5000).cast("float")) \
               .withColumn("claim_amount", col("payer_share_amount").cast("float")) \
               .withColumn("claim_hash", md5(concat_ws("|", col("max_serviced_date"), col("payer_share_amount"), col("provider_license_key")))) \
               .withColumn("history_hashes", array(lit("dummy_hash")))
               
    output_path = dirs['data_output_dir'] / "abt_financial.parquet"
    df_abt.write.mode("overwrite").parquet(str(output_path))
```
**Explanation:**
*   **Simulation**: We don't have a real `Plan_ID` in the raw CSV, so we randomly assign one (`rand() * 3`). We also simulate `ytd_spend`.
*   **Hashing**:
    *   `concat_ws("|", ...)`: Joins columns into a string: `"2023-01-01|100.0|PROV1"`.
    *   `md5(...)`: Hashes it. This creates a unique fingerprint for the claim.
*   **History**: We create a dummy `history_hashes` array. In production, this would be a join with the historical claims table.

## 3.4 Step 3: Feature Engineering (`steps/03_feature_engineering.py`)

This step applies the math.

```python
from medins_ml_utils.features import create_financial_approval_features

def run_step():
    # ... load ABT and Artifacts ...
    
    # Feature Engineering
    df_features = create_financial_approval_features(df, artifacts)
    
    # Output
    output_path = dirs['data_output_dir'] / "features_financial.parquet"
    df_features.write.mode("overwrite").parquet(str(output_path))
```
**Explanation:**
*   **Simplicity**: Notice how short this file is? That's the power of the Hub. All the complex logic (`create_financial_approval_features`) is imported. This file just orchestrates the I/O.

## 3.5 Step 4: Training (`steps/train.py`)

This step builds the brain.

```python
# ... imports ...

def run_step():
    # ... load features ...
    
    # Convert to Pandas for Training
    pdf = df.select("plan_utilization_percent", "claim_amount", "provider_network_tier", "is_duplicate_suspect").toPandas()
    
    # Create Dummy Target (Simulation)
    # Reject if utilization > 90% OR out-of-network (tier 2) OR duplicate
    pdf['is_financially_rejected'] = ((pdf['plan_utilization_percent'] > 0.9) | 
                                      (pdf['provider_network_tier'] > 1) | 
                                      (pdf['is_duplicate_suspect'] == 1)).astype(int)
    
    X = pdf[["plan_utilization_percent", "claim_amount", "provider_network_tier", "is_duplicate_suspect"]]
    y = pdf['is_financially_rejected']
    
    # Train
    model = lgb.LGBMClassifier(**config['model_params'])
    model.fit(X, y)
    
    # Save
    model.booster_.save_model(str(model_path))
```
**Explanation:**
*   **Simulation Target**: Since we don't have real labels, we *define* the ground truth based on rules: "If utilization > 90%, it IS rejected".
*   **Model Learning**: The LightGBM model will now "learn" these rules.
    *   Why not just use the rules? Because in reality, the rules are fuzzy. Maybe 91% utilization is okay for a Gold Plan member? The model learns the exceptions.
*   **`model.fit(X, y)`**: The standard Scikit-Learn style API.

## 3.6 Deployment (`deployment/serve.py`)

This step exposes the model to the world.

```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Inputs: claim_amount, prior_spend, plan_limit
    claim_amount = float(data.get('claim_amount', 0))
    prior_spend = float(data.get('prior_spend', 0))
    plan_limit = float(data.get('plan_limit', 10000.0))
    
    # On-the-fly Feature Engineering
    plan_utilization_percent = (prior_spend + claim_amount) / plan_limit
    
    features = {
        'plan_utilization_percent': plan_utilization_percent,
        # ... other features ...
    }
    
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return jsonify({'financial_rejection_prob': float(pred)})
```
**Explanation:**
*   **Real-time Logic**: In batch training (Spark), we pre-calculated features. In real-time serving, we receive raw numbers (`claim_amount`) and must calculate the ratio `plan_utilization_percent` *inside the request handler*.
*   **Consistency**: We must ensure the formula used here `(a+b)/c` matches the Spark UDF exactly. Ideally, we would use a shared library for this too (the Hub's `online_features` module), but for simple arithmetic, inline is acceptable for v1.

This model is now ready to run. It sits at the front of the pipeline, filtering out bad claims before they reach the expensive clinical models.
