# Chapter 4: The Core Business - Actuarial Pricing

This is the flagship model. Insurance companies live or die by their ability to price risk accurately.

Location: `ml_hub/actuarial_pricing_model_v5`.

## 4.1 The Challenge

We are predicting a continuous variable: **Annualized Cost**.
*   **Range:** $0 to $5,000,000.
*   **Distribution:** Extremely right-skewed (Gamma / Tweedie distribution). Most people cost little; a few cost millions.
*   **Metric:** Mean Absolute Error (MAE). RMSE is bad because squaring a $1M error dominates the loss function.

## 4.2 `steps/01_data_discovery.py`

```python
# ...
def run_step():
    # ...
    # Calculate Risk Weights
    # For every disease code (E11, I10, etc.), calculate avg cost
    
    # Spark SQL Logic (Conceptual):
    # SELECT diagnosis, AVG(cost) / GLOBAL_AVG(cost) as weight
    # FROM claims
    # GROUP BY diagnosis
    
    artifacts = {
        "chronic_condition_cost_weights": {
            "E11": 1.8, # Diabetes type 2
            "I10": 1.2  # Hypertension
        }
    }
    save_artifacts(artifacts, ...)
```
**Explanation:**
*   This step builds the "Actuarial Table". It tells us that a Diabetic patient is expected to cost 1.8x the average member.

## 4.3 `steps/02_build_abt.py`

This is the most complex ETL step in the system.

```python
# ...
    # Flattening the Star Schema
    # Claims (Header) -> Items (Diagnosis) -> CareTeam (Provider)
    
    # Group by Patient
    df_grouped = df.groupBy("Patient_KEY").agg(
        collect_list("cost").alias("claim_costs_list"),
        collect_set("diagnosis").alias("diagnosis_codes_list"),
        collect_list("service_date").alias("service_dates_list")
    )
```
**Explanation:**
*   **`collect_list`**: Transforms multiple rows into a single array.
    *   Row 1: Cost 100
    *   Row 2: Cost 50
    *   **Result**: `[100, 50]`
*   **Why?**: Machine Learning models expect 1 row per entity (Member). We compress the history into arrays so we can extract features from them later.

## 4.4 `steps/03_feature_engineering.py`

This uses `medins_ml_utils.features.create_renewal_vice_champion_features`.

**Key Features:**
1.  **`cost_volatility_iqr`**: `IQR([100, 100, 100]) = 0`. `IQR([50, 5000, 100]) = High`. This measures *stability*. Stable patients are predictable; volatile patients need a risk premium.
2.  **`days_between_visits_std`**: Measures *regularity*.
    *   Regular checkups = Low Std Dev.
    *   Crisis events = High Std Dev.
3.  **`cost_trend_slope`**: Are they getting sicker?
    *   We fit a linear regression line to their cost history. Positive slope = increasing risk.

## 4.5 `steps/05_gradient_boosting.py`

Training the regressor.

```python
    model = lgb.LGBMRegressor(
        objective='regression_l1', # MAE
        n_estimators=1000,
        learning_rate=0.01
    )
    model.fit(X, y)
```
**Explanation:**
*   **`objective='regression_l1'`**: This tells LightGBM to optimize for Mean Absolute Error.
*   **`n_estimators=1000`**: We use many trees with a low learning rate (`0.01`) to prevent overfitting.

## 4.6 The Champion/Challenger Strategy

The model folder is named `actuarial_pricing_model_v5`. Why v5?
*   **v4 (Vice-Champion):** Uses patient history only.
*   **v5 (Champion):** Adds **Provider Risk**.

The code in `create_renewal_champion_features` (in the Hub) adds:
*   `avg_provider_risk_score`: Who is treating this patient? If they see "Dr. Fraud" (identified by the FWA model), their predicted cost goes up because Dr. Fraud overbills.

This connects the Pricing Model to the FWA Model.

---

# Chapter 5: The Shield - FWA Detection

**Fraud, Waste, and Abuse (FWA)** detection is unique because we often lack labels. We don't know who the fraudsters are until we catch them.

Location: `ml_hub/fwa_detection_model_v1`.

## 5.1 Step 1: Unsupervised Learning (`01_unsupervised_model.py`)

```python
from sklearn.ensemble import IsolationForest

def run_step():
    # ...
    # Features: [Avg Cost, Weekend Ratio, Duplicate Ratio]
    X = df_features.toPandas()
    
    iso = IsolationForest(contamination=0.01)
    df['anomaly_score'] = iso.fit_predict(X)
```
**Explanation:**
*   **`IsolationForest`**: This algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value. Anomalies are isolated faster (fewer splits).
*   **`contamination=0.01`**: We assume 1% of providers are outliers.
*   **Output**: A score. -1 is an anomaly.

## 5.2 The Feedback Loop (`02_supervised_retraining.py`)

Once the Special Investigation Unit (SIU) reviews the anomalies, they mark them as `Fraud` (1) or `Clean` (0).

```python
    if len(labels) > 50:
        # We have enough labels to switch to Supervised Learning
        model = lgb.LGBMClassifier()
        model.fit(X, y)
    else:
        # Not enough data, stick to Isolation Forest
        logging.info("Insufficient labels. Keeping Unsupervised model.")
```
**Explanation:**
*   **Cold Start Problem**: Solved by starting Unsupervised and graduating to Supervised.
*   This self-improving loop is the most powerful part of the system.

---

# Chapter 6: Operations - STP & Network Value

## 6.1 STP Automation
**Straight-Through Processing** aims to pay claims without human intervention.
*   **Key Feature**: `has_pre_auth`. If a surgery was pre-approved, we just pay it.
*   **The "Pend" Class**: The model predicts `APPROVE`, `DENY`, or `PEND`. If the model is uncertain (probability < 0.99), it predicts `PEND` and sends it to a human. We prioritize **Precision** over Recall.

## 6.2 Network Value
This model scores providers to negotiate contracts.
*   **Inputs**:
    *   `quality_score` (Clinical outcomes).
    *   `efficiency_score` (Cost per episode).
    *   `fwa_score` (From FWA model).
*   **Output**: Tier 1 (Gold), Tier 2 (Silver), Tier 3 (Bronze).
*   **Usage**: Used in `financial_approval_model` to enforce "Out of Network" penalties for Tier 3 providers.

---

# Epilogue

You have now toured the entire MedIns Machine Learning System.
1.  **Hub (`medins_ml_utils`)**: The shared foundation.
2.  **Financial Approval**: The first gate.
3.  **Pricing**: The core financial engine.
4.  **FWA**: The security system.
5.  **STP**: The automation engine.

By decoupling these concerns into "Spokes" but connecting them via a shared "Hub" and Artifacts, we achieve a system that is both **modular** (easy to maintain) and **integrated** (high business value).
