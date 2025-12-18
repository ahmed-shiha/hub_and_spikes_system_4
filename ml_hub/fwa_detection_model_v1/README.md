# FWA Detection Model v1: Comprehensive Documentation

This document provides a detailed overview of the Fraud, Waste, and Abuse (FWA) Detection Model, a hybrid machine learning system designed to identify and prioritize suspicious provider behavior for investigation.

---

### 1. Primary Business Problem & Model Objective

**Business Problem:** Fraud, Waste, and Abuse (FWA) in healthcare claims is a multi-billion dollar problem. Malicious or negligent providers can exploit the system through various schemes, such as billing for services not rendered ("phantom billing"), billing for more expensive services than were provided ("upcoding"), or performing unnecessary procedures. Static, rule-based systems are often too rigid and can be easily circumvented by determined fraudsters.

**Model Objective:** The objective of this model is to **detect and flag providers exhibiting anomalous or suspicious behavior**. It moves beyond simple rules to identify complex patterns that are indicative of FWA. The system is designed to:
-   **Optimize Investigator Resources:** By providing a prioritized worklist of high-probability FWA suspects, the model allows the Special Investigations Unit (SIU) to focus their efforts where they are most needed.
-   **Adapt to Evolving Fraud Schemes:** Through an Active Learning loop, the model can learn from investigator feedback and adapt to new and emerging fraud patterns.
-   **Reduce Financial Losses:** Proactively identifying FWA helps to stop improper payments before they are made and to recover funds that have been wrongfully paid.

---

### 2. Scope & Boundaries

**In-Scope:**
-   **Provider-level Analysis:** The model's primary unit of analysis is the **provider**. It aggregates claim data to build a profile of each provider's behavior over time.
-   **Hybrid Anomaly Detection & Classification:** The system employs a two-stage approach:
    1.  **Unsupervised Anomaly Detection:** An Isolation Forest model is used to identify providers whose billing patterns are statistical outliers compared to their peers. This is especially useful for "cold start" scenarios where no labeled fraud data exists.
    2.  **Supervised Classification:** Once investigators have labeled a sufficient number of providers, a LightGBM classification model is trained to directly predict the likelihood of FWA.
-   **Active Learning Loop:** The model is designed to be retrained regularly with feedback from investigators, allowing it to continuously improve its accuracy.

**Out-of-Scope:**
-   **Member-centric Fraud:** The model does not focus on fraud perpetrated by members (e.g., identity theft, doctor shopping).
-   **Collusion/Network Fraud:** This version of the model analyzes providers in isolation. It is not designed to detect sophisticated collusion rings involving multiple providers and members (which would require graph-based analysis).
-   **Real-time Claim Blocking:** The model operates in a batch mode to generate a worklist. It is not an in-line system that blocks individual claims in real-time.

---

### 3. Key Performance Indicators (KPIs) & Success Metrics

Success is measured by the model's ability to effectively direct investigator attention to actual cases of FWA.

**Business KPIs:**
-   **Investigator Hit Rate:** The percentage of providers on the model-generated worklist who are confirmed as FWA after a full investigation. A high hit rate is the primary measure of the model's success.
-   **Recovery Amount:** The total dollar amount recovered as a result of investigations initiated by the model.
-   **Investigator Efficiency:** The time it takes for an investigator to close a case. By providing better leads, the model should reduce this time.

**Technical Metrics:**
-   **Precision:** Of all the providers the model flags, what percentage are actually FWA? High precision is critical to maintain investigator trust and avoid "alert fatigue."
-   **Recall:** Of all the true FWA providers in the system, what percentage did the model successfully identify?
-   **F1-Score:** The harmonic mean of precision and recall, providing a balanced measure of the model's accuracy, especially in the context of imbalanced classes.

---

### 4. Data Dictionary & Lineage

The model aggregates claim data to the provider level. The final ABT is one-row-per-provider.

| Column Name                     | Data Type     | Source System             | Description                                                                                              |
| ------------------------------- | ------------- | ------------------------- | -------------------------------------------------------------------------------------------------------- |
| `provider_license_key`          | `String`      | Provider Master           | Unique identifier for the provider.                                                                      |
| `avg_cost_per_claim`            | `Double`      | Claims History            | The provider's average billed amount per claim.                                                          |
| `std_cost_per_claim`            | `Double`      | Claims History            | The standard deviation of the provider's claim amounts. High values indicate inconsistent billing.     |
| `weekend_billing_ratio`         | `Double`      | Claims History            | The percentage of a provider's claims that are for services rendered on a weekend.                       |
| `high_cost_claim_ratio`         | `Double`      | Claims History            | The percentage of a provider's claims that exceed a predefined high-cost threshold (e.g., $5,000).        |
| `is_fraud`                      | `Integer`     | Investigator Feedback     | **Target Variable (for Supervised Model):** 1 if the provider was confirmed as FWA, 0 otherwise.         |

**Data Lineage:**
1.  **Source Systems (Simulated):** Raw claims data (`F_Claim_Header`) is the primary input.
2.  **ETL (`01_unsupervised_model.py`):** A PySpark job aggregates the raw claims data to the provider level, creating the features listed above.
3.  **Investigator Feedback Loop (Simulated):** In a real system, the worklist generated by the unsupervised model would be sent to investigators. Their findings (the `is_fraud` labels) would be stored in a case management system. The `02_supervised_retraining.py` script simulates this by applying logic to the anomaly scores.

---

### 5. Feature Engineering & Logic

The features are designed to capture common FWA indicators.

-   **`avg_cost_per_claim` & `std_cost_per_claim`:**
    -   **Purpose:** To identify providers who are consistently more expensive than their peers or who have erratic billing patterns.
    -   **Considerations:** These features should ideally be normalized by provider specialty and location to ensure fair comparisons.

-   **`weekend_billing_ratio`:**
    -   **Purpose:** To flag providers who may be billing for services that did not occur ("phantom billing"), as clinics are typically closed on weekends.
    -   **Considerations:** Some specialties (e.g., emergency medicine) are expected to have high weekend activity. This feature is most effective when compared to a peer group.

-   **`high_cost_claim_ratio`:**
    -   **Purpose:** To detect "upcoding," where a provider bills for a more expensive service than the one that was actually performed.
    -   **Considerations:** The high-cost threshold needs to be set carefully and may vary by specialty.

---

### 6. Model vs. Rules-Based Logic

This model is designed to augment, not replace, a rules-based FWA system.

-   **Rules-Based Logic:** Traditional FWA systems use hard-coded rules, such as:
    -   `ALERT IF procedure_code == 'X' AND patient_gender == 'Male'` (e.g., billing for childbirth for a male patient).
    -   `ALERT IF billing_location != service_location`.
    These rules are excellent for catching clear, logical impossibilities.

-   **Machine Learning Model:** The ML model excels at detecting patterns that are not logically impossible but are statistically improbable.
    -   The unsupervised model finds providers who are "outliers" compared to their peers, even if no single bill is definitively fraudulent.
    -   The supervised model learns complex, non-linear combinations of features that are associated with confirmed cases of fraud from the past.

---

### 7. Stakeholders & Downstream Consumers

-   **Special Investigations Unit (SIU):** The primary users. The model's worklist is their starting point for investigations.
-   **Data Science Team:** Responsible for the model's lifecycle.
-   **Case Management System:** The downstream system that ingests the worklist and tracks the progress of investigations.

---

### 8. Risk Assessment & Mitigation

-   **Risk of False Accusations:**
    -   **Risk:** The model could incorrectly flag a legitimate provider, causing reputational damage and wasting investigator time.
    -   **Mitigation:** The model is tuned for **high precision**. The goal is to ensure that when the model makes an accusation, it is highly likely to be correct. The output is always treated as a "recommendation for investigation," not a final verdict.

-   **Risk of "Gaming" the System:**
    -   **Risk:** Sophisticated fraudsters could learn the model's features and adjust their billing patterns to stay just below the detection threshold.
    -   **Mitigation:** The Active Learning loop is a key defense. As investigators uncover new schemes, the model is retrained on this new data, allowing it to adapt. The feature set should also be expanded over time.

-   **Risk of Insufficient Labeled Data:**
    -   **Risk:** The supervised model cannot be trained without a sufficient number of high-quality labels from investigators.
    -   **Mitigation:** The system has a built-in "safety fallback." If there are not enough labeled cases, the system relies solely on the unsupervised Isolation Forest model, ensuring that it can still provide value by flagging statistical anomalies.

---

### 9. Scenario Walkthroughs & Examples

**Scenario 1: The Outlier Dentist (Unsupervised Detection)**
-   **Provider Data:** A dentist who consistently bills for 5-10 times the number of services per patient compared to other dentists in the same city.
-   **Model Prediction:** The Isolation Forest model would assign a high anomaly score to this provider because their behavior is a significant statistical outlier.
-   **Outcome:** The provider is placed on the FWA worklist. An investigator would then perform a detailed audit of the provider's claims.

**Scenario 2: The Upcoding Specialist (Supervised Detection)**
-   **Provider Data:** A specialist who has a slightly higher `avg_cost_per_claim` and a slightly higher `high_cost_claim_ratio`.
-   **Model Prediction:** Neither feature alone is a strong enough signal. However, the supervised LightGBM model, trained on past examples, has learned that this specific combination of features is highly predictive of upcoding fraud. It assigns a high probability of FWA.
-   **Outcome:** The provider is prioritized on the worklist for investigation.

---

### 10. Operational & Implementation Details

-   **Pipeline:** The two-step pipeline (unsupervised -> supervised) is designed to be run on a regular schedule (e.g., monthly).
-   **Model Registration:** The trained supervised model is registered in MLflow, allowing for versioning and easy rollback if a new model underperforms.
-   **Feedback Loop:** A critical component of the operational process is ensuring that the labels generated by the SIU are fed back into the model's training dataset for the next retraining cycle.
-   **Future Enhancements:** Future versions could incorporate graph-based features to detect collusion, or Natural Language Processing (NLP) on clinical notes to find mismatches between the documented service and the billed service.
