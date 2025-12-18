# Chapter 5: The Shield - FWA Detection

**Fraud, Waste, and Abuse (FWA)** is a multi-headed monster. It's not just "Bad Doctors". It's also "Professional Patients" (Addicts/Resellers) and "Impossible Claims".

Location: `ml_hub/fwa_detection_model_v1`.

We have evolved the FWA system into a **Tri-Layer Defense**.

## 5.1 Layer 1: Provider FWA (`01_provider_fwa.py`)
This targets the source of 80% of fraud: The Provider.
*   **Logic:** Aggregates thousands of claims to find statistical outliers.
*   **Features:**
    *   `weekend_billing_ratio`: Are they working when they shouldn't be?
    *   `high_cost_claim_ratio`: Do they upcode everything?
*   **Algorithm:** Isolation Forest.

## 5.2 Layer 2: Claim FWA (`02_claim_fwa.py`)
This targets individual transaction anomalies. Even a good doctor might make a typo or try to slip one bad claim through.
*   **Logic:** Analyzes each claim in isolation.
*   **Features:**
    *   `is_round_amount`: Checks if `amount % 100 == 0`. Real medical bills are rarely round numbers ($500.00). They are messy ($493.21).
    *   `starts_with_nine`: (Benford's Law). Amounts starting with 9 (e.g., $990) are often used to stay just under a $1000 approval threshold.
*   **Output:** A list of specific `Claim_IDs` to audit.

## 5.3 Layer 3: Patient FWA (`03_patient_fwa.py`)
This targets "Doctor Shopping" and Identity Theft.
*   **Logic:** Aggregates a patient's history.
*   **Features:**
    *   `unique_providers_30d`: Did this patient see 10 different GPs in 1 month? That is highly suspicious (Drug seeking).
    *   `claim_velocity_7d`: 5 MRIs in a week? Physically impossible.
*   **Output:** A list of `Patient_KEYs` to investigate.

## 5.4 The Feedback Loop (`02_supervised_retraining.py`)

(This remains the same: supervised learning kicks in once we have labels).
