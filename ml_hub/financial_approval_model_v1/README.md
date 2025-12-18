# Financial Approval Model v1: Comprehensive Documentation

This document provides a detailed overview of the Financial Approval Model, a machine learning system designed to predict and flag claims that are likely to be rejected for financial or administrative reasons.

---

### 1. Primary Business Problem & Model Objective

**Business Problem:** A significant portion of health insurance claims are rejected not because of medical necessity, but due to administrative and financial issues. These include claims filed against expired policies, services exceeding plan limits, or submissions from out-of-network providers. Manually reviewing every claim for these issues is operationally expensive and time-consuming.

**Model Objective:** The primary objective of this model is to **proactively identify claims that are likely to be rejected for financial reasons**. By flagging these claims early, the system allows human auditors to prioritize their work on high-risk cases, leading to:
-   **Increased Operational Efficiency:** Reduces the manual effort required to review claims that are clearly within financial limits.
-   **Faster Adjudication:** Speeds up the payment of legitimate claims by quickly filtering out problematic ones.
-   **Reduced Financial Risk:** Minimizes payouts for services that are not covered under the member's plan.

The model serves as a **predictive filter**, not a final decision-maker. It produces a risk score that helps determine whether a claim should be sent for "straight-through processing" (STP) or escalated for human review.

---

### 2. Scope & Boundaries

**In-Scope:**
-   **Financial & Administrative Checks:** The model's predictions are strictly based on financial and administrative data. This includes:
    -   Member's plan utilization (e.g., year-to-date spending vs. plan limits).
    -   Provider's network status (in-network, out-of-network).
    -   Claim amount relative to historical norms.
    -   Suspicion of duplicate claims.
-   **Pre-Adjudication Risk Scoring:** The model is designed to run *before* the formal adjudication process. It provides a recommendation; it does not perform the adjudication itself.
-   **Simulated Environment:** This version of the model operates on simulated data that mirrors a real-world production environment.

**Out-of-Scope:**
-   **Medical Necessity:** The model **does not** evaluate whether a service was medically necessary. This is handled by a separate "Medical Approval Model."
-   **Fraud, Waste, and Abuse (FWA):** While some features (like duplicate claim detection) may overlap with FWA, this model is not designed to be a comprehensive FWA detection system. A dedicated "FWA Detection Model" handles sophisticated fraud pattern analysis.
-   **Real-time Policy Verification:** The model assumes that the policy information provided in the analytical base table (ABT) is up-to-date. It does not perform live lookups against a policy database.

---

### 3. Key Performance Indicators (KPIs) & Success Metrics

Success for this model is measured by its ability to accurately identify financially risky claims without impeding the processing of legitimate ones.

**Business KPIs:**
-   **Straight-Through Processing (STP) Rate:** The percentage of claims that are processed automatically without human review. A key goal is to *increase* this rate for low-risk claims.
-   **Auditor Efficiency:** Measured by the percentage of claims flagged by the model that are ultimately rejected by human auditors. A high "hit rate" indicates the model is effective at prioritizing work.
-   **Improper Payment Rate:** The percentage of financially invalid claims that are mistakenly paid. The model aims to *reduce* this rate.

**Technical Metrics:**
-   **Precision:** Of all the claims the model flags as high-risk, what percentage are actually rejected? High precision is critical to ensure that auditors' time is spent on valuable work.
-   **Recall:** Of all the claims that *should* have been rejected, what percentage did the model correctly flag? High recall is important to minimize financial losses.
-   **Area Under the Precision-Recall Curve (AUC-PR):** As the classes (rejected vs. approved) are imbalanced, AUC-PR is a more informative metric than standard ROC AUC. It summarizes the trade-off between precision and recall across different probability thresholds.

---

### 4. Data Dictionary & Lineage

The model is trained on an **Analytical Base Table (ABT)** that aggregates information from multiple simulated source systems.

| Column Name              | Data Type     | Source System (Simulated) | Description                                                                                               |
| ------------------------ | ------------- | ------------------------- | --------------------------------------------------------------------------------------------------------- |
| `claim_id`               | `String`      | Claims System             | Unique identifier for the insurance claim.                                                                |
| `member_id`              | `String`      | Member Database           | Unique identifier for the patient.                                                                        |
| `plan_id`                | `String`      | Policy Database           | Identifier for the member's insurance plan.                                                               |
| `claim_amount`           | `Double`      | Claims System             | The total amount billed on the claim.                                                                     |
| `ytd_spend`              | `Double`      | Claims History DB         | The member's total accumulated medical spending for the current year, *prior* to this claim.                  |
| `provider_license_key`   | `String`      | Provider Master           | Unique identifier for the healthcare provider who rendered the service.                                   |
| `claim_hash`             | `String`      | ETL Process               | A generated hash of key claim fields (e.g., date, amount, provider) to detect potential duplicates.     |
| `history_hashes`         | `Array<String>`| Claims History DB         | An array of claim hashes for the member's recent claims, used for duplicate checking.                     |
| `is_rejected_financial`  | `Integer`     | Adjudication System       | **Target Variable:** 1 if the claim was rejected for financial reasons, 0 otherwise.                        |

**Data Lineage:**
1.  **Source Systems (Simulated):** Raw data tables (e.g., `F_Claim_Header`, `D_Member`, `D_Plan`) are simulated as CSVs in a central data directory.
2.  **ETL (`02_build_abt.py`):** A PySpark script reads these tables, joins them, and aggregates data to create the one-row-per-claim ABT. This step also generates the `claim_hash` and `history_hashes`.
3.  **Feature Engineering (`03_feature_engineering.py`):** The ABT is then processed by the `create_financial_approval_features` function from the `medins_ml_utils` library to generate the final features for training.

---

### 5. Feature Engineering & Logic

All features are generated using PySpark for scalability. The core logic resides in `medins_ml_utils.features.py`.

-   **`plan_utilization_percent`**
    -   **Logic:** `(ytd_spend + claim_amount) / plan_limit`
    -   **Purpose:** This is the most critical feature. It calculates what percentage of the member's annual plan limit will be consumed if the current claim is approved.
    -   **Considerations:** The `plan_limit` is retrieved from a `plan_limits` artifact, which is a dictionary lookup. In a production system, this would come from a dedicated plan management database.

-   **`provider_network_tier`**
    -   **Logic:** A numerical tier is assigned based on the provider's status: `1` for in-network, `2` for out-of-network, and `3` for blacklisted.
    -   **Purpose:** Claims from out-of-network or blacklisted providers are often subject to different financial rules or outright rejection.
    -   **Considerations:** This feature relies on a `provider_master` artifact. In a real system, this would be a constantly updated provider database.

-   **`is_duplicate_suspect`**
    -   **Logic:** `1` if the `claim_hash` of the current claim exists in the `history_hashes` array, `0` otherwise.
    -   **Purpose:** To flag potential duplicate submissions, which are a common source of administrative rejections.
    -   **Considerations:** The effectiveness of this feature depends on the quality and look-back window of the `history_hashes`.

---

### 6. Model vs. Rules-Based Logic

This system is a **hybrid** of rules-based logic and machine learning.

-   **Rules-Based Logic (Hard Rules):** Certain conditions lead to an automatic rejection without any ML prediction. For example:
    -   `IF plan_utilization_percent > 1.0 THEN REJECT` (The claim exceeds the hard limit).
    -   `IF provider_network_tier == 3 THEN REJECT` (The provider is on a blacklist).
    These hard rules are typically implemented in the adjudication system *after* the model's prediction, but they are documented here for completeness.

-   **Machine Learning Model (Soft Rules):** The LightGBM model handles the "gray areas" where the decision is not clear-cut. It learns from historical data to identify complex patterns. For example:
    -   A claim that brings utilization to 95% from an out-of-network provider might be riskier than a similar claim from an in-network provider.
    -   A small claim amount might be low-risk even if other factors are borderline.

The model's output (a probability score) is used to segment claims:
-   **Low Score (e.g., < 0.1):** High confidence; send for Straight-Through Processing.
-   **Medium Score (e.g., 0.1 to 0.7):** Borderline; send to a junior auditor.
-   **High Score (e.g., > 0.7):** High suspicion; escalate to a senior auditor.

---

### 7. Stakeholders & Downstream Consumers

-   **Claims Adjudication Team:** The primary users of the model's output. They rely on the risk scores to prioritize their work queues.
-   **Data Science Team:** Responsible for monitoring the model's performance, retraining it, and developing new features.
-   **Provider Network Management:** Insights from the `provider_network_tier` feature can be used to identify providers who frequently cause issues.
-   **Downstream Systems:** The adjudication platform is the main downstream system that consumes the model's risk score.

---

### 8. Risk Assessment & Mitigation

-   **Risk of Bias:**
    -   **Risk:** The model could inadvertently learn to penalize certain provider types or geographical locations if the training data contains historical biases.
    -   **Mitigation:** The model's performance is monitored across different provider specialties, locations, and plan types to detect any significant disparities. The `validation` scripts include bias checks.

-   **Risk of Model Drift:**
    -   **Risk:** The financial behavior of members or the structure of insurance plans may change over time, causing the model's performance to degrade.
    -   **Mitigation:** The model is retrained on a regular schedule (e.g., quarterly) with fresh data. Monitoring systems are in place to track key feature distributions and model KPIs, with alerts for significant drift.

-   **Risk of Over-Reliance:**
    -   **Risk:** The claims team might become overly reliant on the model and stop performing random audits of "low-risk" claims.
    -   **Mitigation:** A small, random percentage of claims flagged as low-risk are still sent to human auditors to ensure the model is not developing a blind spot.

---

### 9. Scenario Walkthroughs & Examples

**Scenario 1: Clear Approval**
-   **Claim Data:**
    -   `claim_amount`: $150.00
    -   `ytd_spend`: $2,000.00
    -   `plan_limit`: $10,000.00
    -   `provider_network_tier`: 1 (In-Network)
-   **Feature Values:**
    -   `plan_utilization_percent`: (2000 + 150) / 10000 = 21.5%
-   **Model Prediction:** The model would likely predict a very low risk score (e.g., 0.05).
-   **Outcome:** The claim is sent for Straight-Through Processing and paid automatically.

**Scenario 2: Clear Rejection (Hard Rule)**
-   **Claim Data:**
    -   `claim_amount`: $500.00
    -   `ytd_spend`: $9,800.00
    -   `plan_limit`: $10,000.00
-   **Feature Values:**
    -   `plan_utilization_percent`: (9800 + 500) / 10000 = 103%
-   **Model Prediction:** The model would predict a high risk, but it's irrelevant.
-   **Outcome:** The adjudication system's hard rule for exceeding the plan limit would automatically reject the claim before it even gets to an auditor.

**Scenario 3: Borderline Case for ML**
-   **Claim Data:**
    -   `claim_amount`: $1,200.00
    -   `ytd_spend`: $8,500.00
    -   `plan_limit`: $10,000.00
    -   `provider_network_tier`: 2 (Out-of-Network)
-   **Feature Values:**
    -   `plan_utilization_percent`: (8500 + 1200) / 10000 = 97%
-   **Model Prediction:** The model sees a high utilization percentage *and* an out-of-network provider. It has learned that this combination is risky and predicts a high score (e.g., 0.85).
-   **Outcome:** The claim is flagged and escalated to a senior auditor, who will investigate whether the out-of-network service is covered at such a high utilization level.

---

### 10. Operational & Implementation Details

-   **Training Pipeline:** The model is trained using a sequence of PySpark scripts (`01_` to `train.py`) that are orchestrated by a workflow manager (e.g., Airflow, Dagster).
-   **Deployment:** The trained LightGBM model is saved as an artifact. A FastAPI-based REST API wraps the model for real-time predictions.
-   **Monitoring:** Model performance (Precision, Recall) and feature distributions are logged to MLflow and monitored for drift.
-   **Retraining Cadence:** The model is scheduled to be retrained quarterly to capture the latest data patterns. The "Champion/Challenger" validation framework is used to ensure that a new "challenger" model only replaces the current "champion" if it demonstrates superior performance on a hold-out dataset.
