# Reference: Actuarial Pricing Model (v5)

## 1. Business Context
**The Problem:** Traditional insurance pricing relies on broad "Actuarial Tables" (e.g., "All Males aged 30-35 pay $X"). This fails to account for individual health status, leading to:
*   **Overpricing** healthy members (who then leave -> Churn).
*   **Underpricing** sick members (leading to financial loss).

**The Solution:** A Machine Learning model that predicts the exact **Annualized Cost** (Payer Share) for a specific member for the upcoming year.

## 2. Model Workflow

### Phase A: Onboarding (New Business)
*   **Scenario:** A new person applies for insurance. We have zero claims history.
*   **Input Data:** Age, Gender, City, Self-Reported Conditions (checked boxes on a form).
*   **Model:** `pricing-onboarding`.
*   **Logic:** Relies heavily on demographic baselines and a "Self-Reported Risk Score".

### Phase B: Renewal (Existing Business)
*   **Scenario:** A member has been with us for 1+ years. We have rich claims history.
*   **Input Data:** All of the above + ICD-10 Diagnoses, Visit Frequency, Provider usage.
*   **Model Strategy (Champion/Challenger):**
    *   **Vice-Champion:** Uses patient history but ignores *who* they visited. Good baseline.
    *   **Champion:** Adds "Provider Risk". If a patient visits expensive/fraudulent doctors, their predicted cost goes up.

## 3. Detailed Feature Reference

### 3.1 `self_reported_chronic_score` (Onboarding)
*   **Logic:** Weighted sum of conditions.
*   **Weights:** Derived from historical data (Step 01).
*   **Example:**
    *   Diabetes Weight: 1.8 (Costs 80% more than average).
    *   Hypertension Weight: 1.2.
    *   Member reports both. Score = (1.8 + 1.2) / 2 = 1.5.
*   **Business Insight:** Allows pricing to adjust immediately for known risks without waiting for a claim.

### 3.2 `data_driven_risk_score` (Renewal)
*   **Logic:** Same as above, but sources conditions from **Verified ICD-10 Codes** in claims, not self-reports.
*   **Why?** Self-reports are often understated. Claims data is the "ground truth" of health status.

### 3.3 `cost_volatility_iqr`
*   **Logic:** Interquartile Range (75th percentile - 25th percentile) of a member's claim costs.
*   **Example:**
    *   Member A claims: [100, 100, 100]. IQR = 0. (Stable).
    *   Member B claims: [50, 5000, 100]. IQR = High. (Volatile).
*   **Business Insight:** Volatile members are harder to price and carry higher financial risk. We add a premium for this uncertainty.

### 3.4 `days_between_visits_std`
*   **Logic:** Standard Deviation of time gaps between doctor visits.
*   **Example:**
    *   Regular checkups (every 30 days): Low Std Dev. -> Well managed.
    *   Erratic visits (2 days, then 200 days, then 5 days): High Std Dev. -> Crisis-driven care.
*   **Business Insight:** Crisis-driven care is usually more expensive than preventative maintenance.

### 3.5 `cost_trend_slope`
*   **Logic:** Linear regression slope of claim costs over time.
*   **Example:** Slope > 0 means costs are rising (getting sicker). Slope < 0 means recovering.
*   **Business Insight:** We price based on *future* trajectory, not just past average.

## 4. Technical Implementation Steps

### Step 01: Data Discovery & Artifacts
*   **Goal:** Calculate the "Risk Weights" for diseases.
*   **Process:**
    1.  Load Training Data (Spark).
    2.  Filter for members with "Diabetes" (ICD-10 E11).
    3.  Calculate their Average Cost vs Global Average.
    4.  Save this ratio (e.g., 1.8) to `data_driven_artifacts.json`.
*   **Why Spark?** We must scan millions of claim lines to get statistically significant averages for rare diseases.

### Step 02: Build ABT (Analytical Base Table)
*   **Goal:** Flatten the relational database (Star Schema) into a single flat table.
*   **Process:**
    1.  **Join:** `F_Claim_Header` (Cost) + `F_Claim_Item` (Diagnosis) + `F_CLAIM_CARETEAM` (Provider).
    2.  **Aggregate (Spark):** Group by `Patient_KEY`.
        *   `collect_list(cost)` -> `[100, 200, 50]`
        *   `collect_set(diagnosis)` -> `['E11', 'I10']`
    3.  **Store:** Save as `abt_renewal.parquet`.

### Step 03: Feature Engineering
*   **Goal:** Transform raw lists into mathematical features.
*   **Process:** Apply UDFs (User Defined Functions) from `medins_ml_utils`.
    *   Input: `[100, 200, 50]`
    *   Function: `calculate_iqr`
    *   Output: `75.0`

### Step 05: Training
*   **Goal:** Learn the mapping from Features -> Annual Cost.
*   **Algorithm:** LightGBM Regressor (`objective='regression_l1'` aka Mean Absolute Error).
*   **Why MAE?** Insurance data has massive outliers (million-dollar claims). RMSE (Root Mean Square) punishes outliers too heavily, skewing prices for everyone else. MAE is more robust for median pricing.
