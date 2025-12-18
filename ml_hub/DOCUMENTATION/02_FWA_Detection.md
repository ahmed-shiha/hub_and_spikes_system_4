# Reference: FWA Detection Model (v1)

## 1. Business Context
**The Problem:** Fraud, Waste, and Abuse (FWA) accounts for 3-10% of healthcare spend.
*   **Fraud:** Intentional deception (e.g., billing for services not rendered).
*   **Waste:** Inefficiency (e.g., ordering redundant MRIs).
*   **Abuse:** Stretching the rules (e.g., "Upcoding" a simple checkup to a complex consultation).

**The Solution:** An AI system that acts as a "Radar" for the Special Investigation Unit (SIU).

## 2. The "Active Learning" Workflow
Fraudsters adapt. Static rules ("If Bill > $5000") are easily bypassed. We need a system that learns.

1.  **Cold Start (Unsupervised):**
    *   We don't know what new fraud looks like.
    *   We use **Isolation Forest** to find statistical outliers.
    *   *Result:* A "Worklist" of the 50 most suspicious providers.

2.  **The Labeling Bay (Human-in-the-Loop):**
    *   Investigators investigate the Worklist.
    *   They tag providers: "Innocent (Just expensive)" or "Guilty (Fraud)".

3.  **Retraining (Supervised):**
    *   Once we have labels, we train a **Classifier**.
    *   This classifier learns the specific patterns of *confirmed* fraud, improving precision over time.

## 3. Detailed Feature Reference

### 3.1 `weekend_billing_ratio`
*   **Logic:** (Claims on Fri/Sat/Sun) / (Total Claims).
*   **Business Insight:** Most legitimate clinics are closed on weekends. A provider billing 30% of their volume on Sundays is highly suspicious (Phantom Billing).

### 3.2 `high_cost_claim_ratio`
*   **Logic:** % of claims > 5,000 SAR.
*   **Business Insight:** Detects "Upcoding". If a General Practitioner (GP) has 50% high-cost claims, they are likely billing for services they didn't perform or exaggerating severity.

### 3.3 `std_cost_per_claim`
*   **Logic:** Standard Deviation of billing amounts.
*   **Business Insight:**
    *   **Fraud:** Often uniform. A fraudster might bill "500 SAR" exactly for every patient to stay under the radar. (Std Dev ~ 0).
    *   **Real Life:** Messy. Different patients need different care. (Std Dev > 0).
    *   *Note:* The model learns this non-linear relationship.

## 4. Technical Implementation Steps

### Step 01: Unsupervised Scanning (Spark)
*   **Scale:** Scans millions of claims across thousands of providers.
*   **Aggregation:**
    *   Group by `provider_license_key`.
    *   Calculate metrics (Avg Cost, Weekend Ratio).
*   **Modeling:** `sklearn.ensemble.IsolationForest`.
    *   `contamination=0.01`: We assume ~1% of providers are anomalous.
    *   Output: `anomaly_score`. Lower is more anomalous.

### Step 02: Supervised Retraining
*   **Trigger:** Runs weekly.
*   **Input:** The `fwa_worklist.csv` + Investigator Labels (`is_fraud`).
*   **Logic:**
    *   If enough "Fraud" labels exist (>50), train a LightGBM Classifier.
    *   If not, fallback to the Unsupervised score.
