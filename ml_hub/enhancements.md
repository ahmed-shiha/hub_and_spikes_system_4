# ML Hub Enhancement Roadmap

This document outlines potential future enhancements for the models and infrastructure within the ML Hub. It is intended to be a living document that captures ideas for improvement and guides future development efforts.

---

## 1. Model-Specific Enhancements

### A. Actuarial Pricing Model (`actuarial_pricing_model_v5`)

This model is already quite sophisticated, but it can be improved by incorporating more granular data and more advanced modeling techniques.

**New Features:**
-   **Provider Network Features:**
    -   **`avg_provider_tier`:** The average value tier (from the `network_value_model`) of the providers a member visits. This could be a powerful predictor of cost.
    -   **`specialist_visit_ratio`:** The ratio of visits to specialists vs. general practitioners.
-   **Pharmacy Data:**
    -   **`prescription_count`:** The number of prescriptions a member has filled.
    -   **`adherence_score`:** A score that measures how consistently a member takes their prescribed medications.
-   **Temporal Features:**
    -   **`months_since_last_visit`:** A measure of how recently a member has engaged with the healthcare system.
    -   **`claim_frequency_trend`:** A feature that captures whether a member's visit frequency is increasing or decreasing.

**Alternative Modeling Approaches:**
-   **Deep Learning for Claims Sequencing:** Instead of aggregating claims into a single row, a Recurrent Neural Network (RNN) or a Transformer model could be used to model the sequence of claims. This could capture temporal patterns that are missed by the current approach.
-   **Quantile Regression:** Instead of predicting the mean expected cost, a quantile regression model could be used to predict a range of possible costs (e.g., the 75th and 95th percentiles). This would provide a better understanding of the potential upside risk for each member.

**Minor Enhancements:**
-   **Automated Feature Selection:** The current `04_feature_selection.py` script is a good start, but a more automated and robust feature selection process could be implemented.
-   **Hyperparameter Tuning:** A more extensive hyperparameter tuning process for the LightGBM model could yield further performance improvements.

### B. FWA Detection Model (`fwa_detection_model_v1`)

The hybrid unsupervised/supervised approach is a strong foundation, but it could be enhanced by incorporating more sophisticated fraud detection techniques.

**New Features:**
-   **Graph-based Features:**
    -   **`provider_collusion_score`:** A score that measures the likelihood that a provider is colluding with other providers or with members. This would require building a graph of the provider-member network.
    -   **`shared_attribute_count`:** The number of other providers who share the same address, phone number, or bank account.
-   **Procedure-level Features:**
    -   **`service_uniqueness_score`:** A score that measures how unusual a particular procedure is for a provider of a given specialty.
    -   **`upcoding_indicator`:** A feature that specifically looks for patterns of billing for more expensive procedures than are medically necessary.

**Alternative Modeling Approaches:**
-   **Graph Neural Networks (GNNs):** A GNN could be used to model the entire provider-member network, allowing it to learn complex collusion patterns directly.
-   **Autoencoders:** An autoencoder could be used as a more sophisticated unsupervised anomaly detector, potentially outperforming the Isolation Forest.

### C. STP Automation Model (`stp_automation_model_v1`)

This model's primary goal is precision. Enhancements should focus on safely increasing the STP rate without compromising accuracy.

**New Features:**
-   **Natural Language Processing (NLP) on Clinical Notes:**
    -   **`note_complexity_score`:** A score that measures the complexity of the clinical notes associated with a claim.
    -   **`billing_code_mismatch`:** A feature that flags discrepancies between the services mentioned in the clinical notes and the codes billed on the claim.
-   **Provider History:**
    -   **`provider_pend_rate`:** The historical percentage of a provider's claims that have been sent to `PEND`.

**Alternative Modeling Approaches:**
-   **Hierarchical Models:** A hierarchical model could be used to model the fact that claims are nested within members, who are nested within providers. This could improve the model's ability to learn from sparse data.

### D. Network Value Model (`network_value_model_v1`)

The definition of "value" in this model could be significantly enhanced by incorporating direct measures of quality.

**New Features:**
-   **Clinical Outcome Data:**
    -   **`readmission_rate`:** The percentage of a provider's patients who are readmitted to the hospital within 30 days of a procedure.
    -   **`complication_rate`:** The percentage of a provider's patients who experience complications.
-   **Patient Satisfaction Scores:**
    -   **`avg_patient_rating`:** The average satisfaction rating given by a provider's patients.

**Alternative Modeling Approaches:**
-   **Gaussian Mixture Models (GMMs):** A GMM could be used as a more flexible clustering algorithm that can capture clusters of different shapes and sizes.

### E. Member Churn Model (`member_churn_model_v1`)

This model could be improved by incorporating more direct measures of member sentiment and by looking at external market factors.

**New Features:**
-   **Call Center Log Analysis (NLP):**
    -   **`sentiment_score`:** The sentiment (positive, negative, neutral) of a member's interactions with the call center.
    -   **`complaint_count`:** The number of formal complaints a member has filed.
-   **External Market Data:**
    -   **`competitor_premium_delta`:** The difference between the member's premium and the average premium for a similar plan from a competitor.

**Alternative Modeling Approaches:**
-   **Survival Analysis:** Instead of predicting a binary `Will Churn` outcome, a survival analysis model could be used to predict the *time until* a member is likely to churn. This would provide a more nuanced view of churn risk.

---
## 2. The Importance of Raw Data and Artifacts

A robust MLOps practice recognizes that the outputs of the machine learning pipeline are not just the final models, but also the data artifacts generated at each step. Storing and governing these intermediate artifacts is critical for debugging, reproducibility, and advanced analytics.

**Key Artifacts to Capture:**

-   **`01_data_discovery` Artifacts:**
    -   **Column-wise Statistics:** For each raw table, save summary statistics (mean, median, min, max, cardinality, null count) for every column. This is invaluable for detecting data drift.
    -   **Correlation Matrices:** A correlation matrix for the raw data can reveal relationships that might not be immediately obvious.

-   **`02_build_abt` Artifacts:**
    -   **The Analytical Base Table (ABT) Itself:** The ABT is arguably the most important artifact. It should be versioned and stored, as it represents the single source of truth for the model at a specific point in time.
    -   **Intermediate Join Logs:** Logs that detail the number of rows dropped or kept at each join step in the ETL process are crucial for debugging data quality issues.

-   **`03_feature_engineering` Artifacts:**
    -   **The Featured ABT:** The final table with all engineered features should be saved. This allows data scientists to perform offline analysis and exploration without having to re-run the entire feature engineering pipeline.

**How to Use These Artifacts:**

-   **Data Drift Detection:** By comparing the column-wise statistics from the current run to a historical baseline, we can automatically detect significant shifts in the input data (e.g., the average claim amount has suddenly increased by 20%).
-   **Reproducibility:** If a model from six months ago needs to be investigated, having the versioned ABT and featured ABT from that time makes it possible to perfectly reproduce the training conditions.
-   **Advanced Analytics:** Business analysts and data scientists can query the historical, versioned ABTs to answer complex questions about how the member or provider population has changed over time, without needing to understand the complexities of the upstream source systems.
-   **Faster Iteration:** When developing new features, data scientists can start from the versioned ABT instead of having to re-run the entire, often time-consuming, ETL process from the raw source data.
