# System Architecture & Technical Plan

## Overview
The system follows a "Hub & Spoke" architecture designed for scalability, reusability, and governance. The core ("The Hub") provides shared utilities, data connectors, and feature engineering logic. The "Spokes" are independent model projects that consume the Hub.

## Design Goals
1.  **Scalability**: Handle "millions of rows" by migrating data processing from Pandas to **PySpark**.
2.  **Consistency**: Centralized feature logic in `medins_ml_utils`.
3.  **Completeness**: Full lifecycle support (Discovery -> ETL -> Features -> Training -> Deployment).

## Components

### 1. The Hub (`medins_ml_utils`)
*   **Technology**: Python, PySpark.
*   **Role**: Shared Library.
*   **Modules**:
    *   `data_connectors`: Spark-based readers for CSV, Parquet, SQL (via JDBC).
    *   `features`: PySpark native implementations of feature engineering logic. avoiding UDFs for performance.
    *   `utils`: Spark session management, config loading.

### 2. The Spokes (Model Projects)
Each spoke follows a standard pipeline structure (steps 01-05+).

#### A. Actuarial Pricing Model (`actuarial_pricing_model_v5`)
*   **Goal**: Predict annualized cost for members.
*   **Pipeline**:
    *   `01_data_discovery`: Spark-based profiling.
    *   `02_build_abt`: Join tables using Spark (efficient shuffling).
    *   **Features**:
        *   *Onboarding*: Demographic, Self-reported conditions.
        *   *Renewal*: Claims history, Diagnoses (Chronic conditions), Provider utilization.
    *   `04_feature_selection`: (New) Correlation/Importance analysis.
    *   `05_training`: LightGBM (fed by Spark -> Pandas for memory-fitting datasets).

#### B. FWA Detection Model (`fwa_detection_model_v1`)
*   **Goal**: Detect Fraud, Waste, and Abuse.
*   **Technique**: Unsupervised (Anomaly Detection) + Supervised (Active Learning).
*   **Data**: Claims, Provider behavior.

#### C. STP Automation Model (`stp_automation_model_v1`)
*   **Goal**: Auto-adjudicate claims (Straight-Through Processing).
*   **Technique**: High-precision Classification.
*   **Integration**: Feeds "friction" metrics to Churn model.

#### D. Network Value Model (`network_value_model_v1`)
*   **Goal**: Segment providers (Value Tiers).
*   **Technique**: Clustering.

#### E. Member Churn Model (`member_churn_model_v1`)
*   **Goal**: Predict member attrition.
*   **Features**: Demographics, Claims friction (from STP), Cost changes (from Pricing).

## Data Flow & Optimization
*   **Raw Data**: Ingested via Spark.
*   **Transformations**: Lazy evaluation in Spark.
*   **Materialization**: Intermediate ABTs saved as Parquet (compressed, columnar) instead of CSV.
*   **Training**:
    *   *Small/Medium Data (<10GB)*: `toPandas()` -> LightGBM.
    *   *Big Data (>10GB)*: Spark MLlib or SynapseML LightGBM. (We will implement the standard `toPandas()` approach for this iteration as it covers most "millions of rows" use cases where the *aggregated* training set is smaller than the raw transactional data).

## Future Enhancements
*   **Feature Store**: Manage feature versioning (Feast or Databricks Feature Store).
*   **Model Monitoring**: Drift detection on Spark streams.
*   **Real-time Serving**: REST API wrapping the model (FastAPI/Flask) - already present, needs hardening.
