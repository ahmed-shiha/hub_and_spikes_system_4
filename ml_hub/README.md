# The ML Hub: A "Hub & Spoke" MLOps Architecture (v5 - Fully Implemented)

This repository demonstrates a professional "Hub & Spoke" MLOps architecture.

## 1. The Hub: `medins_ml_utils`

This is a central, installable Python library (`setup.py`) that contains all shared, reusable code.

-   **Data Connectors:** Standardized functions to connect to production data sources.
-   **Feature Logic:** The single source of truth for feature engineering.
-   **Utilities:** Common helper functions.

It is versioned, tested, and published to a private package repository.

## 2. The Spokes: Model-Specific Projects

Each ML model gets its own project folder, which acts as a "spoke". These spokes are consumers of the shared hub library.

-   **`actuarial_pricing_model_v5/`:** A sophisticated batch-prediction model with a complete, multi-step training pipeline, automated validation, Champion/Challenger deployment strategy, and full observability.
-   **`fwa_detection_model_v1/`:** A real-time prediction model demonstrating an Active Learning training loop, online feature fetching, and its own governed MLOps lifecycle.
-   **`stp_automation_model_v1/`:** A high-precision, real-time classification model for automating claim adjudication with a human-over-the-loop safety net.
-   **`network_value_model_v1/`:** An unsupervised clustering model that segments providers into value tiers, consuming insights from the FWA model.
-   **`member_churn_model_v1/`:** A classification model to predict member churn, consuming operational metrics from the STP system as "friction features".

This architecture promotes code reuse, consistency, and independent model development and deployment.
