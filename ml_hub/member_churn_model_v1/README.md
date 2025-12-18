# Member Churn Model v1 (Documentation)

## 1. Idea & Goal
Losing members (Churn) is costly. Members often leave due to "friction" (denied claims, high costs).
**Goal:** Predict which members are at risk of leaving to trigger retention campaigns.

## 2. Methodology (The "Way")
**Binary Classification** (LightGBM).
This model depends heavily on **Upstream Signals** from the Pricing and STP models.

## 3. Data & Inputs
*   **Source:** `abt_renewal.parquet` (Output of Pricing Model).
*   **STP Integration:** Consumes `manual_adjudication_rate` (delay friction).
*   **Features:**
    *   `out_of_pocket_ratio`: Cost friction.
    *   `premium_increase_percentage`: Price shock friction.
    *   `pre_auth_denial_rate`: Clinical friction.

## 4. Pipeline Scripts
*   **`steps/train.py`**:
    1.  **Data Loading (Spark):** Reads the pre-computed ABT from Pricing Model.
    2.  **Feature Engineering:** Calculates ratios and deltas.
    3.  **Training:** LightGBM classifier.

## 5. Edge Cases
*   **Missing History:** New members (policy duration < 1 year) have different churn drivers.
*   **Dependency Failure:** If Pricing Model fails, Churn Model cannot run (DAG dependency).

## 6. Future Improvements
*   **Sentiment Analysis:** Incorporate Call Center logs.
*   **Competitor Pricing:** Include market rate data.
