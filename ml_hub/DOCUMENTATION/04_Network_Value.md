# Reference: Network Value Model (v1)

## 1. Business Context
MedIns contracts with thousands of providers. We need to categorize them to create tiered networks (e.g., "Gold Network" plans only access high-value doctors).

**Value Equation:**
`Value = (Quality + Efficiency) / Cost`

## 2. Methodology
We use **Unsupervised Clustering (K-Means)**.
Since "Quality" is hard to define universally, we let the data find natural groupings of providers.

## 3. Detailed Feature Reference

### 3.1 `provider_fraud_score`
*   **Source:** The FWA Model.
*   **Logic:** A provider might be "Cheap" (Low Cost), but if they are "Fraudulent", their Value is zero.
*   **Impact:** Providers with high fraud scores are pushed into the lowest tier (Bronze/Excluded), regardless of their efficiency.

### 3.2 `claim_count` (Volume)
*   **Insight:** High volume providers are strategic partners. Even if slightly expensive, we can't afford to lose the biggest hospital in the city. They often form their own "Key Account" cluster.

## 4. Technical Implementation
1.  **Spark Aggregation:** Aggregates millions of claim lines to calculate `avg_cost` per provider.
2.  **Integration:** Joins with the output of the FWA pipeline.
3.  **K-Means:**
    *   `k=3` (Gold, Silver, Bronze).
    *   **Scaling:** We use `StandardScaler` because K-Means is distance-based. A volume of 10,000 and a cost of $100 are on different scales.
