# MedIns ML System Documentation

## Overview
This repository contains the **Hub & Spoke** Machine Learning architecture for MedIns. It is designed to be:
*   **Scalable:** Uses **PySpark** for data processing, handling millions of claims.
*   **Modular:** Centralized logic in `medins_ml_utils`, independent model "spokes".
*   **Production-Ready:** Includes MLOps practices (MLflow, CI/CD, Containerization).

## Architecture

### 1. The Hub: `medins_ml_utils`
*   **Purpose:** Single source of truth for feature engineering and data access.
*   **Tech:** Python, PySpark.
*   **Key Modules:**
    *   `data_connectors`: Unified Spark-based access to Data Warehouse (simulated CSV/Parquet).
    *   `features`: Standardized feature logic (e.g., `calculate_self_reported_score`).

### 2. The Spokes
Each spoke is a self-contained ML project.

| Model | Type | Goal | Tech Stack |
| :--- | :--- | :--- | :--- |
| **Actuarial Pricing** | Regression | Predict Member Annual Cost | Spark, LightGBM |
| **FWA Detection** | Unsupervised/Supervised | Detect Fraudulent Providers | Spark, IsolationForest, LightGBM |
| **STP Automation** | Classification | Auto-adjudicate Claims | Spark, LightGBM |
| **Network Value** | Clustering | Segment Providers | Spark, K-Means |
| **Member Churn** | Classification | Predict Attrition | Spark, LightGBM |

## Data Flow
1.  **Raw Data** (Claims, Members) -> Ingested by **Spark**.
2.  **Pricing Model** creates `abt_renewal.parquet` (Member-level features).
3.  **Churn Model** consumes `abt_renewal.parquet`.
4.  **STP Model** & **FWA Model** generate signals (Fraud Score, Friction Rate).
5.  **Network Model** consumes Fraud Scores.

## Execution Guide
1.  **Install Hub:** `pip install -e ml_hub/medins_ml_utils`
2.  **Run Pricing Pipeline:**
    *   `python ml_hub/actuarial_pricing_model_v5/steps/01_data_discovery.py`
    *   `python ml_hub/actuarial_pricing_model_v5/steps/02_build_abt.py`
    *   `...`
3.  **Run Other Models:** Respective `steps/` scripts.

## Scaling Strategy
*   **ETL:** All heavy joins and aggregations are performed in PySpark.
*   **Training:** Aggregated datasets (e.g., Member-level, Provider-level) are collected to Pandas for LightGBM training. This fits the "millions of rows -> thousands of entities" pattern common in insurance.
