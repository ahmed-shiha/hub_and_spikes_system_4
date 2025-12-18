# MedIns Machine Learning System: Architecture & Design Reference

## 1. Executive Summary
This system is a centralized Machine Learning platform designed for a Health Insurance provider ("MedIns"). It addresses the challenge of scaling data science operations from "laptop-scale" to "enterprise-scale".

The core philosophy is **"Hub & Spoke"**:
*   **The Hub (`medins_ml_utils`)**: A centralized, version-controlled library containing all shared logic (data access, feature engineering definitions, utility functions). This ensures that a feature like "Chronic Condition Score" is calculated exactly the same way in the Pricing Model as it is in the Churn Model.
*   **The Spokes**: Independent ML projects (Pricing, FWA, STP, etc.) that import the Hub. Each spoke manages its own training pipeline, configuration, and deployment lifecycle.

## 2. Technical Architecture

### 2.1 The "Millions of Rows" Challenge
Insurance data (Claims) grows rapidly. A single year can generate tens of millions of claim lines. Traditional tools like Pandas run into memory limits (OOM) on standard training nodes.

**The Solution: Hybrid Processing**
We employ a "Big Data ETL, Small Data Training" pattern:
1.  **Distributed Processing (PySpark)**:
    *   Used for the heavy lifting: Joining massive tables (Claims + Members + Providers), aggregating transactional rows into entity-level summaries, and calculating complex features.
    *   Output: Intermediate Parquet files (columnar, compressed).
2.  **In-Memory Training (LightGBM/Pandas)**:
    *   Once data is aggregated (e.g., to 1 million unique Members), it fits comfortably in memory (e.g., < 1GB RAM).
    *   We collect the featured dataset from Spark to Pandas for final model training using Scikit-Learn and LightGBM.

### 2.2 Data Governance & Connectors
To prevent "CSV hell" (data scattered across random folders), all data access is governed by `medins_ml_utils.data_connectors`.
*   **Code:** `get_data(path, tables, spark_session)`
*   **Logic:** In production, this connects to a Secure Data Warehouse (Snowflake/BigQuery). In this simulation, it strictly reads from a central governed raw data folder.

## 3. The Ecosystem of Models
The models are not isolated; they feed into each other to create a "Flywheel of Value".

1.  **Financial Approval**: Runs first on a claim to check administrative validity (e.g., plan limits).
2.  **STP Automation**: Runs second (if approved financially). Checks medical necessity.
3.  **FWA Detection**: Runs on providers. It generates a "Fraud Score".
3.  **Network Value**: Consumes the "Fraud Score" to penalize risky providers when calculating their Value Tier.
4.  **Actuarial Pricing**: Predicts member cost.
5.  **Member Churn**: Consumes "Friction" signals from STP and "Cost" signals from Pricing to predict if a member will leave.

## 4. MLOps Lifecycle
Each model follows a standardized lifecycle:
1.  **Discovery (Step 01):** Profile data, calculate global statistics (Artifacts) to prevent train-serving skew.
2.  **ETL (Step 02):** Build Analytical Base Tables (ABT) using Spark.
3.  **Feature Engineering (Step 03):** Apply logic from the Hub.
4.  **Training (Step 05):** Train, Validate, and Register to MLflow.
5.  **Deployment:** Containerize (Docker) and expose via REST API.
