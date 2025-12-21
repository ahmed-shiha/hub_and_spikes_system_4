# Actuarial Pricing Model v5: Comprehensive Documentation

This document provides a detailed overview of the Actuarial Pricing Model, a machine learning system designed to predict the annualized healthcare costs for insurance members.

---

### 1. Primary Business Problem & Model Objective

**Business Problem:** Traditional actuarial pricing relies on broad demographic tables (age, gender, location) to set insurance premiums. This approach is often inaccurate at an individual level, leading to two primary issues:
-   **Adverse Selection:** Healthier individuals may find the premiums too high for the value they receive, causing them to leave the insurance pool.
-   **Underpricing Risk:** Individuals with significant health risks may be charged premiums that do not adequately cover their expected medical costs, leading to financial losses for the insurer.

**Model Objective:** The objective of this model is to move beyond static tables and **predict a personalized annualized cost for each member**. By leveraging individual medical history, chronic conditions, and healthcare utilization patterns, the model aims to:
-   **Improve Pricing Accuracy:** Set premiums that more accurately reflect an individual's risk profile.
-   **Reduce Financial Risk:** Ensure the insurer remains solvent by pricing policies sustainably.
-   **Enable Fairer Premiums:** Offer more competitive pricing to lower-risk individuals.

The model's output is a direct prediction of the member's expected cost for the upcoming year, which serves as a primary input for the premium calculation process.

---

### 2. Scope & Boundaries

**In-Scope:**
-   **Annualized Cost Prediction:** The model predicts the total expected medical cost for a member over a 12-month period.
-   **Onboarding vs. Renewal:** The system includes two distinct model variants:
    -   **Onboarding Model:** For new members with no claims history, using only demographic and self-reported health data.
    -   **Renewal Model:** For existing members with at least one year of claims history, leveraging detailed medical and utilization data.
-   **Batch Processing:** The model is designed to run in a batch process (e.g., nightly or weekly) to score the entire member population.

**Out-of-Scope:**
-   **Premium Calculation:** The model predicts the *cost*, not the final *premium*. The premium is calculated by a separate downstream system that adds administrative loads, profit margins, and other business adjustments to the model's cost prediction.
-   **Real-time Quoting:** This is not a real-time system for quoting new policies. The onboarding model is used to price new members who have already signed up.
-   **Catastrophic Event Prediction:** The model is trained on historical data and is not designed to predict the costs of rare, catastrophic "black swan" health events.

---

### 3. Key Performance Indicators (KPIs) & Success Metrics

Success is measured by the accuracy of the cost predictions and the financial health of the insurance pool.

**Business KPIs:**
-   **Loss Ratio:** The ratio of claims paid out to premiums earned. A primary goal is to stabilize and optimize this ratio.
-   **Member Retention/Churn:** Particularly for low-risk members. Fairer pricing should lead to better retention of healthy members.
-   **Market Competitiveness:** The ability to offer competitive premiums based on accurate risk assessment.

**Technical Metrics:**
-   **Mean Absolute Error (MAE):** The primary regression metric used to evaluate the model. It represents the average dollar amount of error in the cost predictions.
-   **Root Mean Squared Error (RMSE):** Also used to evaluate the model. RMSE penalizes large errors more heavily than MAE, making it useful for identifying models that produce wildly inaccurate predictions.
-   **R-squared (R²):** Measures the proportion of the variance in the actual costs that is predictable from the model's features.

---

### 4. Data Dictionary & Lineage

The model is trained on a "one-row-per-member" ABT aggregated from various source systems.

| Column Name                     | Data Type     | Source System             | Description                                                                                              |
| ------------------------------- | ------------- | ------------------------- | -------------------------------------------------------------------------------------------------------- |
| `Patient_KEY`                   | `String`      | Member Database           | Unique identifier for the member.                                                                        |
| `age`                           | `Integer`     | Member Database           | Member's age at the time of scoring.                                                                     |
| `gender`                        | `String`      | Member Database           | Member's gender.                                                                                         |
| `self_reported_conditions`      | `Array<String>`| Onboarding Forms          | List of chronic conditions self-reported by new members.                                                 |
| `diagnosis_codes_list`          | `Array<String>`| Claims History            | List of all ICD-10 diagnosis codes from the member's claims in the past year.                            |
| `claim_costs_list`              | `Array<Double>` | Claims History            | List of costs for each of the member's claims.                                                           |
| `service_dates_list`            | `Array<Date>`   | Claims History            | List of service dates for all claims.                                                                    |
| `practitioner_specialty_list`   | `Array<String>`| Provider Master           | List of specialties for all providers the member has visited.                                            |
| `providers_visited_list`        | `Array<String>`| Claims History            | List of unique provider IDs the member has visited.                                                      |
| `annualized_cost`               | `Double`      | Claims History            | **Target Variable:** The total cost of all claims for the member in the target year.                       |

**Data Lineage:**
1.  **Source Systems (Simulated):** Raw tables like `F_Claim_Header`, `F_Claim_Item`, and `D_Patient` are simulated as CSVs.
2.  **ETL (`02_build_abt.py`):** A heavy-lifting PySpark script joins and aggregates "millions of rows" of claims data into the one-row-per-member ABT.
3.  **Feature Engineering (`03_feature_engineering.py`):** The `medins_ml_utils` library applies standardized feature logic to the ABT.

---

### 5. Feature Engineering & Logic

The model uses a rich set of features to capture different aspects of a member's risk profile.

-   **`self_reported_chronic_score` (Onboarding):**
    -   **Logic:** A weighted sum based on self-reported conditions. The weights are derived from the historical costs associated with each condition.
    -   **Purpose:** To estimate risk for new members without a claims history.

-   **`data_driven_risk_score` (Renewal):**
    -   **Logic:** A more accurate risk score calculated from the member's actual diagnosis history. The logic maps thousands of ICD-10 codes to a smaller set of chronic conditions, which are then weighted.
    -   **Purpose:** The primary driver of the renewal model's predictions.

-   **`cost_volatility_iqr`:**
    -   **Logic:** The interquartile range (75th percentile - 25th percentile) of a member's claim costs.
    -   **Purpose:** To measure the stability of a member's healthcare spending. High volatility might indicate an unstable health condition.

-   **`cost_trend_slope`:**
    -   **Logic:** The slope of a linear regression line fitted to the member's claim costs over time.
    -   **Purpose:** To capture whether a member's health is improving (negative slope) or deteriorating (positive slope).

-   **`care_team_complexity`:**
    -   **Logic:** The number of unique practitioner specialties a member has visited.
    -   **Purpose:** A proxy for the complexity of a member's health conditions. A high number may indicate a patient with multiple co-morbidities.

-   **`avg_provider_risk_score`:**
    -   **Logic:** A credibility-weighted average of the historical cost-efficiency of the providers a member has visited.
    -   **Purpose:** To capture the risk associated with a member's choice of healthcare providers.

---

### 6. Model vs. Rules-Based Logic

This model primarily serves as an input to a larger pricing engine, which may contain business rules.

-   **Rules-Based Logic:** The pricing engine applies certain hard-coded rules, such as:
    -   **State-mandated Rate Caps:** Premiums cannot exceed certain state-regulated limits.
    -   **Age Banding:** Premiums are often constrained within certain age-based bands.
    These rules are applied *after* the model's prediction.

-   **Machine Learning Model:** The ML model's role is to provide the most accurate possible cost prediction, which is then fed into this rules-based engine. The model is flexible and can be updated to reflect new trends in healthcare data without changing the hard-coded business rules.

---

### 7. Stakeholders & Downstream Consumers

-   **Actuarial & Underwriting Teams:** The primary consumers of the model's output. They use the cost predictions to set premiums.
-   **Product & Strategy Teams:** Use insights from the model to design new health plans and understand the risk profile of the member base.
-   **Data Science Team:** Manages the model's lifecycle, from development to monitoring and retraining.
-   **Pricing Engine:** The downstream software system that takes the model's cost prediction and applies business rules to generate the final premium.

---

### 8. Risk Assessment & Mitigation

-   **Risk of Unfair Bias:**
    -   **Risk:** The model could learn to associate higher costs with protected classes (e.g., specific genders, ethnicities, or locations) in a way that is not causally justified.
    -   **Mitigation:** The model's predictions are validated for fairness across different demographic segments. The `validation` scripts include bias disparity checks. Features that are highly correlated with protected classes are carefully reviewed.

-   **Risk of Overfitting:**
    -   **Risk:** The model could learn the noise in the training data too well, leading to poor performance on new, unseen members.
    -   **Mitigation:** The model is trained using cross-validation, and a hold-out test set is used to get a final, unbiased estimate of its performance. Regularization techniques in LightGBM are also used to prevent overfitting.

-   **Risk of Data Leakage:**
    -   **Risk:** Information from the future (the target variable) could inadvertently leak into the training data, leading to an overly optimistic performance estimate.
    -   **Mitigation:** The training pipeline is carefully designed to be point-in-time correct. For example, artifacts like cost weights are calculated *only* on the training set (`01_data_discovery.py`) before being applied to the test set.

---

### 9. Scenario Walkthroughs & Examples

**Scenario 1: New, Healthy Member (Onboarding)**
-   **Data:**
    -   `age`: 28
    -   `gender`: Female
    -   `self_reported_conditions`: [] (None)
-   **Feature Values:**
    -   `self_reported_chronic_score`: 0.0
-   **Model Prediction:** The model would predict a low annualized cost, based primarily on the member's young age and lack of pre-existing conditions.

**Scenario 2: Existing Member with Chronic Condition (Renewal)**
-   **Data:**
    -   `age`: 55
    -   `diagnosis_codes_list`: ['E11.9', 'I10', 'Z79.4'] (Type 2 Diabetes, Hypertension, long-term insulin use)
    -   `cost_trend_slope`: 0.15 (Slightly increasing costs over the past year)
-   **Feature Values:**
    -   `data_driven_risk_score`: A high value due to multiple chronic conditions.
-   **Model Prediction:** The model would predict a significantly higher annualized cost, reflecting the ongoing costs of managing diabetes and hypertension.

**Scenario 3: Member with a Recent, Expensive Surgery (Renewal)**
-   **Data:**
    -   `age`: 42
    -   `claim_costs_list`: [150.0, 200.0, 85000.0, 250.0] (Reflecting a major surgery)
-   **Feature Values:**
    -   `cost_volatility_iqr`: A very high value.
-   **Model Prediction:** The model has learned that a single, high-cost event does not necessarily mean the high costs will continue. It might predict a cost that is higher than average but much lower than the previous year's total, assuming the surgery was a one-time event.

---

### 10. Operational & Implementation Details

-   **Pipeline Scripts:** The model is built by a sequence of PySpark and Python scripts, as detailed in the original documentation.
-   **Champion/Challenger Deployment:** This model uses a Champion/Challenger approach. Any new "challenger" model must outperform the current "champion" on a hold-out dataset before it can be promoted to production. This ensures that the pricing accuracy is always improving.
-   **Observability:** The `monitoring` subfolder contains scripts for tracking the model's performance and data distributions over time, which is crucial for a high-impact system like pricing.
-   **Retraining Cadence:** The model is retrained annually, ahead of the new plan year, to incorporate the latest full year of claims data.
