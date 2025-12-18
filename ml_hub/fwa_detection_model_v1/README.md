# FWA Detection Model v1 (Documentation)

## 1. Idea & Goal
Fraud, Waste, and Abuse (FWA) costs billions. This model detects anomalous **Provider** behavior. Unlike static rule engines, it adapts to new fraud patterns using an **Active Learning** loop.

**Goal:** Identify providers submitting suspicious claims (e.g., upcoding, phantom billing) with high precision to optimize investigator time.

## 2. Methodology (The "Way")
We employ a hybrid **Unsupervised + Supervised** approach:
1.  **Unsupervised (Cold Start):** Isolation Forest detects statistical outliers (anomalies) in provider billing patterns. These form a "Worklist".
2.  **Human-in-the-Loop:** Investigators review the worklist and label providers as "Fraud" or "Not Fraud".
3.  **Supervised (Retraining):** Once enough labels exist, a LightGBM classifier is trained to predict fraud directly.

## 3. Data & Inputs
*   **`F_Claim_Header`**: Aggregated by `provider_license_key`.
*   **Input Features:**
    *   `avg_cost_per_claim`: Is the provider significantly more expensive than peers?
    *   `std_cost_per_claim`: Do they bill inconsistent amounts?
    *   `weekend_billing_ratio`: Do they bill excessively on weekends (phantom visits)?
    *   `high_cost_claim_ratio`: % of claims exceeding a threshold (e.g., 5000 SAR).

## 4. Pipeline Scripts

### A. `01_unsupervised_model.py` (PySpark + Sklearn)
*   **Data Prep:** Uses PySpark to aggregate millions of claims into a `provider_features` table.
*   **Modeling:** `sklearn.ensemble.IsolationForest`.
*   **Output:** `fwa_worklist.csv` (Providers with high anomaly scores).

### B. `02_supervised_retraining.py` (Pandas + LightGBM)
*   **Data Prep:** Reads the labeled worklist (simulated labeling logic included).
*   **Modeling:** `lgb.LGBMClassifier`.
*   **Validation:** Checks if class balance allows for valid training.
*   **Output:** MLflow registered model `fwa-detection-model`.

## 5. Edge Cases & Handling
*   **Low Data Volume:** If < 5 providers, pipeline skips gracefully.
*   **Class Imbalance:** If investigator labels are all "Not Fraud", the supervised step aborts, relying solely on the unsupervised outlier detector (safety fallback).
*   **Spark-to-Pandas:** Aggregation happens in Spark; only the small provider-level summary (thousands of rows) is moved to Pandas for training.

## 6. Future Improvements
*   **Graph Neural Networks (GNN):** To detect collusion rings between providers and members.
*   **Network Analysis:** Detect providers sharing the same bank accounts or phone numbers.
