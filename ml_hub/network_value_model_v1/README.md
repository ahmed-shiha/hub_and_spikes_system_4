# Network Value Model v1: Comprehensive Documentation

This document provides a detailed overview of the Network Value Model, an unsupervised machine learning system designed to segment healthcare providers into distinct value tiers.

---

### 1. Primary Business Problem & Model Objective

**Business Problem:** The cost and quality of healthcare are not uniform across all providers. Some providers achieve better patient outcomes at a lower cost, while others may be inefficient, overly expensive, or even fraudulent. Simply having a large network of providers is not enough; the network must be optimized for value. The business lacks a data-driven way to systematically evaluate and compare providers.

**Model Objective:** The objective of this model is to **segment all providers into "Value Tiers" (e.g., Gold, Silver, Bronze)** based on their historical performance. This segmentation provides actionable intelligence that can be used to:
-   **Optimize Network Contracting:** Prioritize contracts with high-value providers and negotiate more aggressively with low-value ones.
-   **Guide Member Choice:** Steer members towards higher-value providers through plan design and search tools.
-   **Inform Provider Relations:** Identify providers who may need support or education to improve their efficiency.

This is an **unsupervised** model, meaning it discovers natural groupings of providers based on data, rather than predicting a predefined label.

---

### 2. Scope & Boundaries

**In-Scope:**
-   **Provider Segmentation:** The model's sole purpose is to cluster providers based on a combination of cost, volume, and risk metrics.
-   **Unsupervised Learning:** It uses K-Means clustering to identify these segments. The number of clusters (tiers) is a configurable parameter.
-   **Batch Processing:** The model is run periodically (e.g., quarterly) to refresh the provider segmentations based on the latest claims data.
-   **Integration with FWA Model:** A key aspect of this model is its consumption of the `provider_fraud_score` from the FWA Detection Model, incorporating a measure of risk into the definition of "value."

**Out-of-Scope:**
-   **Quality of Care Metrics:** This version of the model defines "value" primarily in terms of cost-efficiency and risk. It does **not** include direct measures of clinical quality, such as patient outcomes, re-admission rates, or complication rates. Incorporating such metrics is a planned future enhancement.
-   **Individual Physician Performance:** The model analyzes providers at the level of the billing entity (e.g., a hospital or a clinic, identified by `provider_license_key`), not individual doctors within that entity.

---

### 3. Key Performance Indicators (KPIs) & Success Metrics

Success for an unsupervised model is measured by the usefulness and stability of the generated clusters.

**Business KPIs:**
-   **Network Cost Savings:** Over time, shifting member utilization towards "Gold" tier providers should lead to a reduction in the total cost of care.
-   **"Gold Tier" Utilization Rate:** The percentage of claims that are for services rendered by top-tier providers.
-   **Provider Network Composition:** The percentage of providers in the network who are in the "Gold" tier.

**Technical Metrics:**
-   **Silhouette Score:** A measure of how similar a provider is to its own cluster compared to other clusters. A high score indicates that the clusters are dense and well-separated.
-   **Inertia (Elbow Method):** The sum of squared distances of samples to their closest cluster center. Used to help determine the optimal number of clusters.
-   **Cluster Stability:** The percentage of providers who remain in the same value tier from one model run to the next. High stability is important for business actionability.

---

### 4. Data Dictionary & Lineage

The model is trained on a "one-row-per-provider" ABT.

| Column Name                     | Data Type     | Source System             | Description                                                                                              |
| ------------------------------- | ------------- | ------------------------- | -------------------------------------------------------------------------------------------------------- |
| `provider_license_key`          | `String`      | Provider Master           | Unique identifier for the provider.                                                                      |
| `avg_cost_per_claim`            | `Double`      | Claims History            | The provider's average billed amount per claim, normalized by specialty.                                 |
| `claim_count`                   | `Integer`     | Claims History            | The total number of claims submitted by the provider. A measure of volume.                               |
| `provider_fraud_score`          | `Double`      | FWA Detection Model       | The anomaly or fraud probability score for the provider, taken from the FWA model.                       |
| `value_tier`                    | `String`      | K-Means Clustering        | **Model Output:** The assigned tier for the provider (e.g., "Gold", "Silver", "Bronze").                     |

**Data Lineage:**
1.  **Source Systems:** Aggregates data from `F_Claim_Header` and consumes the output of the `fwa_detection_model_v1`.
2.  **ETL (`train.py`):** A PySpark job aggregates claims data to the provider level to calculate `avg_cost_per_claim` and `claim_count`. It then joins this with the FWA scores.
3.  **Clustering (`train.py`):** The final provider-level ABT is typically small enough to be collected into a Pandas DataFrame, where Scikit-learn's K-Means algorithm is used to perform the clustering.

---

### 5. Feature Engineering & Logic

The features are designed to capture three dimensions of provider value: Efficiency, Volume, and Risk.

-   **`avg_cost_per_claim` (Efficiency):**
    -   **Purpose:** To identify providers who are cost-efficient.
    -   **Considerations:** This is the most important feature. To be meaningful, it must be normalized. A simple approach is to convert each provider's average cost into a Z-score relative to the mean and standard deviation of their specialty.

-   **`claim_count` (Volume):**
    -   **Purpose:** To provide context. A high-volume provider has a larger impact on the network. They may also have more negotiating power.
    -   **Considerations:** This feature needs to be scaled (e.g., using a log transform) before being used in K-Means, as the algorithm is sensitive to the scale of the input features.

-   **`provider_fraud_score` (Risk):**
    -   **Purpose:** To ensure that providers who are cheap but risky are not placed in the top value tier.
    -   **Considerations:** This creates a valuable dependency between the FWA and Network Value models, ensuring a more holistic view of provider performance.

---

### 6. Model vs. Rules-Based Logic

This is a classic use case where machine learning is superior to a rules-based approach.

-   **Rules-Based Logic:** A business user might try to define value tiers with simple rules, such as:
    -   `IF avg_cost < X AND fraud_score < Y THEN 'Gold'`
    This approach is brittle, hard to maintain, and does not discover natural groupings in the data. The thresholds (X, Y) are arbitrary and may not be optimal.

-   **Machine Learning Model:** K-Means automatically finds the optimal cluster centroids based on the multi-dimensional data. It can identify groups of providers with similar trade-offs between cost, volume, and risk that would not be obvious from simple rules. For example, it might identify a "High-volume, Medium-cost" segment that is distinct from a "Low-volume, Low-cost" segment.

---

### 7. Stakeholders & Downstream Consumers

-   **Network Strategy & Contracting Team:** Use the value tiers to make decisions about which providers to include in the network and how to structure their contracts.
-   **Member Services & Digital Team:** Use the tiers to build tools that help members find high-value providers.
-   **Provider Relations Team:** Use the segmentations to identify and engage with providers who could benefit from performance improvement initiatives.

---

### 8. Risk Assessment & Mitigation

-   **Risk of Unstable Segments:**
    -   **Risk:** The cluster assignments could change dramatically with each retraining, making it difficult for the business to act on the results.
    -   **Mitigation:** Cluster stability is a key validation metric. If the clusters are unstable, it may be necessary to use a more robust clustering algorithm or to refine the features.

-   **Risk of Misinterpretation:**
    -   **Risk:** Business users might misinterpret the meaning of the tiers. For example, they might assume that "Gold" means "highest clinical quality," when the model is primarily based on cost-efficiency.
    -   **Mitigation:** Clear and thorough documentation is the primary mitigation. The labels for the tiers should be chosen carefully (e.g., "Most Cost-Efficient" might be better than "Gold").

-   **Risk of Penalizing Legitimate Outliers:**
    -   **Risk:** A provider might be a high-cost outlier for a legitimate reason (e.g., they are a world-renowned specialist who handles only the most complex cases).
    -   **Mitigation:** The clustering should be performed on a specialty-by-specialty basis. Additionally, any provider flagged for placement in the lowest tier should be manually reviewed before any action is taken.

---

### 9. Scenario Walkthroughs & Examples

**Cluster 1: "Gold Tier" (High Value)**
-   **Profile:** Low `avg_cost_per_claim`, low `provider_fraud_score`, and high `claim_count`.
-   **Interpretation:** These are the ideal providers. They are cost-efficient, low-risk, and handle a significant volume of patients.
-   **Business Action:** Ensure these providers are prominently featured in member search tools. Renew their contracts with favorable terms.

**Cluster 2: "Bronze Tier" (Low Value)**
-   **Profile:** High `avg_cost_per_claim` and/or high `provider_fraud_score`.
-   **Interpretation:** These are high-risk or inefficient providers who are driving up the cost of care.
-   **Business Action:** These providers are candidates for contract renegotiation, auditing, or, in extreme cases, removal from the network.

**Cluster 3: "Specialist Niche"**
-   **Profile:** High `avg_cost_per_claim` but low `provider_fraud_score` and low `claim_count`.
-   **Interpretation:** The model might identify a small cluster of legitimate, highly-specialized providers who are expensive but necessary for the network.
-   **Business Action:** No adverse action. The model has correctly identified them as a distinct group that should not be penalized for their high costs.

---

### 10. Operational & Implementation Details

-   **Pipeline:** The model is trained via a single PySpark/Python script (`train.py`).
-   **Output:** The primary output is a CSV or Parquet file that maps each `provider_license_key` to its assigned `value_tier`. This file is then loaded into downstream systems.
-   **Cluster Labeling:** After the K-Means algorithm assigns a numerical cluster label (0, 1, 2, etc.), a post-processing step is required to analyze the centroids of each cluster and assign meaningful business labels ("Gold", "Silver", "Bronze"). This step should be repeatable and automated.
-   **Retraining Cadence:** The model should be retrained quarterly to ensure the provider segmentations are based on up-to-date data.
