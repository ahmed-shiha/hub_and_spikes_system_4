# Chapter 9: Deep Dive Data Analysis & Architectural Standardization

We have gained access to the **Raw Data Layer** (`ml_hub/data/raw_data/`). This chapter performs a "maliciously fine-grained" analysis of these tables to uncover hidden value, contradictions, and architectural opportunities.

## 9.1 The Raw Data Ecosystem

### 1. `hidp_v2.csv` (Policy Master)
*   **Content:** `identitynumber`, `policyexpirydate`, `premium`, `classname` (Tier), `gender`.
*   **Value:** **Financial Goldmine.**
    *   **Churn:** `premium` vs `claim_cost` = Profitability.
    *   **Approval:** `policyexpirydate` is the *hardest* check. If `service_date > expiry`, REJECT immediately.
    *   **Medical:** `gender` and `age` are ground truth here (more reliable than claim forms).
*   **Problem:** Columns are lowercase (`gender`) vs TitleCase (`Gender`) in other tables. Requires standardization.

### 2. `f_pre_auth_header.csv` & `f_pre_auth_item.csv`
*   **Content:** Approvals *before* the claim. `disposition`, `adjudication_outcome`, `principal_diagnosis_code`.
*   **Value:** **Medical Ground Truth.**
    *   If Pre-Auth = "Approved", the Claim Medical Probability is 99%.
    *   If Pre-Auth = "Rejected" (e.g., "Refill too soon"), this is a strong FWA signal if they claim it anyway.
*   **Contradiction Risk:** A Pre-Auth might be for "Surgery A" but the Claim is for "Surgery B". The `pre_auth_ref` link is vital.

### 3. `d_product_or_service.csv`
*   **Content:** `service_key`, `activity_type` (Medication, Medical devices), `display` (Name).
*   **Value:** **Medical Necessity.**
    *   `activity_type` = "Medication" requires a Diagnosis that matches the drug indication.
    *   `activity_type` = "Medical devices" (e.g., Pacemaker) requires a specific Surgery procedure.
*   **Model Usage:** `medical_approval_model` should feature `service_type_consistency`.

### 4. `d_adjudication_reason.csv`
*   **Content:** `code` (N-DC-027), `display` ("Vaccinations are not covered").
*   **Value:** **Labeling.** This is the *Target Variable* for our classifiers.
    *   "Not covered" = Financial Denial.
    *   "Inconsistent Diagnosis" = Medical Denial.
    *   We must map these codes to `[FINANCIAL, MEDICAL, FWA]` targets.

## 9.2 Architectural Critique: The "Maturity Gap"

Our current models are uneven:
*   **Mature:** `actuarial_pricing` (Has Artifacts, ABT, Training).
*   **Immature:** `stp_automation` (Hardcoded rules).
*   **Missing:** Standardized **EDA (Exploratory Data Analysis)**.

### The Proposed Standard Pipeline
Every Spoke must implement this interface:
1.  **`00_eda.py`**: Profile the data. Check for nulls, drift, and schema violations. Output: `eda_report.html`.
2.  **`01_discovery.py`**: Calculate Artifacts (Global Stats).
3.  **`02_abt.py`**: Join tables into a flat Feature Vector.
4.  **`03_features.py`**: Apply `medins_ml_utils` logic.
5.  **`04_train.py`**: Model fitting.
6.  **`05_explain.py`**: SHAP values / Feature Importance.

## 9.3 Medical vs Financial Approval: The Split

Based on the raw tables, we refine the boundaries:

| Feature | Source Table | Model | Why? |
| :--- | :--- | :--- | :--- |
| **Policy Active?** | `hidp_v2` | **Financial** | Contractual. Binary. |
| **Premium Paid?** | `hidp_v2` | **Financial** | Financial standing. |
| **Service Covered?** | `d_product_or_service` | **Financial** | Benefit plan design. |
| **Diagnosis Match?** | `f_pre_auth_item` | **Medical** | Clinical logic. |
| **Refill Frequency** | `claim_history` | **Medical** | Clinical safety / FWA. |
| **Gender Match** | `d_patient` | **Medical** | Biological possibility. |

## 9.4 Action Plan
We will upgrade `medical_approval_model_v1` to use `d_product_or_service` and `d_diagnosis_code` from the raw folder, demonstrating the new "Standard Pipeline" by adding `00_eda.py`.
