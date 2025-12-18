# Network Value Model v1 (Documentation)

## 1. Idea & Goal
Not all providers are equal. Some deliver better outcomes at lower costs.
**Goal:** Segment providers into "Value Tiers" (Gold, Silver, Bronze) to guide network contracting and member routing.

## 2. Methodology (The "Way")
**Unsupervised Clustering (K-Means)**. We group providers based on cost-efficiency, volume, and fraud risk.

## 3. Data & Inputs
*   **`F_Claim_Header`**: Aggregated claim costs.
*   **FWA Integration:** Consumes `provider_fraud_score` from the FWA model.
*   **Features:**
    *   `avg_cost_per_claim`: Efficiency metric.
    *   `claim_count`: Volume metric (high volume = better negotiating power).
    *   `provider_fraud_score`: Risk metric.

## 4. Pipeline Scripts
*   **`steps/train.py`**:
    1.  **Spark Aggregation:** Computes provider stats from millions of claims.
    2.  **Pandas/Sklearn:** Performs K-Means clustering on the provider summary.
    3.  **Output:** Assigns a `cluster_label` to each provider.

## 5. Edge Cases
*   **New Providers:** Assigned to a default cluster until enough claims accumulate.
*   **Outliers:** Extremely expensive providers might form their own tiny cluster or skew centroids (handled by Scaling).

## 6. Future Improvements
*   **Outcome-based features:** Re-admission rates, complication rates (requires clinical data).
