# Actuarial Pricing Model v5 (Documentation)

## 1. Idea & Goal
The **Actuarial Pricing Model** determines the fair annual premium for health insurance members. Instead of relying solely on static actuarial tables, this model uses Machine Learning to predict the **Annualized Cost** of a member based on their medical history, chronic conditions, and utilization patterns.

**Goal:** Minimize the difference between predicted cost and actual cost (MAE) to ensure competitive yet sustainable pricing.

## 2. Methodology (The "Way")
We use a **Gradient Boosting (LightGBM)** regressor. The pipeline is designed for "Big Data" scale using **PySpark** for data processing and feature engineering, transitioning to Pandas/LightGBM for the final training on aggregated data.

### Architecture
*   **Hub & Spoke:** Consumes feature logic from `medins_ml_utils`.
*   **Scalability:** PySpark handles millions of claims rows.
*   **Evaluation:** Mean Absolute Error (MAE) on a hold-out test set.

## 3. Data & Inputs
The model ingests raw data from the Data Warehouse (simulated as CSVs/Parquet):
*   **`F_Claim_Header`**: Claim amounts, dates, provider IDs.
*   **`F_Claim_Item`**: Diagnosis codes (ICD-10).
*   **`D_Patient` / `hidp_v2`**: Member demographics (Age, Gender, City).
*   **`F_CLAIM_CARETEAM`**: Providers visited.

## 4. Feature Engineering
All features are generated via `medins_ml_utils` using PySpark.

### A. Onboarding Features (New Members)
Used when claims history is unavailable.
*   **`self_reported_chronic_score`**: Weighted score based on self-declared conditions (Diabetes, Hypertension, etc.) and their associated cost weights derived from historical data.
*   **`age_x_chronic_score`**: Interaction term.

### B. Renewal Features (Existing Members)
*   **`data_driven_risk_score`**: Risk score derived from actual ICD-10 diagnosis history.
*   **`cost_volatility_iqr`**: Interquartile range of claim costs (measures stability).
*   **`days_between_visits_std`**: Irregularity of visits.
*   **`cost_trend_slope`**: Slope of cost over time (is the patient getting sicker?).
*   **`care_team_complexity`**: Count of unique practitioner specialties visited.
*   **`pre_auth_denial_rate`**: % of pre-auths rejected (proxy for friction/inappropriate care).

### C. Provider Features (Champion Model)
*   **`avg_provider_risk_score`**: Average cost-efficiency of providers visited by the member.
*   **`hospital_visit_ratio`**: Ratio of hospital visits vs. clinics.

## 5. Pipeline Scripts
The pipeline is sequential:

1.  **`01_data_discovery.py`**:
    *   **Tool:** PySpark.
    *   **Action:** Profiling data, calculating global artifacts (Chronic Condition Cost Weights) from the training set to prevent leakage.
    *   **Output:** `data_driven_artifacts.json`.

2.  **`02_build_abt.py`**:
    *   **Tool:** PySpark.
    *   **Action:** Joins Claims, Patients, and Diagnoses. Aggregates "millions of rows" into a "one-row-per-member" Analytical Base Table (ABT).
    *   **Output:** `abt_renewal.parquet`, `abt_onboarding.parquet`.

3.  **`03_feature_engineering.py`**:
    *   **Tool:** PySpark.
    *   **Action:** Applies feature functions from `medins_ml_utils` to the ABTs.
    *   **Output:** `abt_renewal_champion_featured.parquet`.

4.  **`04_feature_selection.py`**:
    *   **Tool:** PySpark MLlib.
    *   **Action:** Calculates correlation matrix to identify redundant features or top predictors.

5.  **`05_gradient_boosting.py`**:
    *   **Tool:** LightGBM + MLflow.
    *   **Action:** Trains 3 model variants (Onboarding, Renewal Vice-Champion, Renewal Champion). Logs metrics (MAE) and artifacts to MLflow.

## 6. Edge Cases & Handling
*   **New Members:** Handled by the "Onboarding" model variant which uses only demographic/self-reported data.
*   **Zero Claims:** Members with no claims get a base risk score (0) but demographics still apply.
*   **Null Values:** Handled in Spark (coalesce to 0 or empty lists) and LightGBM (native NaN support).
*   **Large Data:** PySpark ensures we don't OOM during the join/aggregate phase.

## 7. Future Improvements
*   **Deep Learning:** For sequence modeling of claims (LSTM/Transformer) instead of simple aggregation.
*   **Real-time Pricing:** Pricing based on live claim stream (streaming features).
