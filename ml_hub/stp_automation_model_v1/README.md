# STP Automation Model v1 (Documentation)

## 1. Idea & Goal
Straight-Through Processing (STP) automates claim adjudication. Manual review is slow and expensive.
**Goal:** Auto-approve or auto-deny simple claims with >99% precision, sending only complex/ambiguous cases ("PEND") to human adjudicators.

## 2. Methodology (The "Way")
A **Multi-class Classifier** (LightGBM) predicts one of three outcomes: `APPROVE`, `DENY`, or `PEND`.
*   **High Precision Requirement:** We prioritize Precision for `APPROVE` and `DENY` classes. If the model is unsure, it *must* predict `PEND`.

## 3. Data & Inputs
*   **`F_Claim_Header`**: Claim metadata.
*   **`F_Pre_Auth_Header`**: Does the claim have a prior approval?
*   **Features:**
    *   `net_amount`: High value claims are riskier.
    *   `has_pre_auth`: Claims with pre-auth are safer.
    *   `service_is_covered`: Binary flag from policy rules.
    *   `is_complex_diag`: Binary flag for complex diseases.

## 4. Pipeline Scripts
*   **`steps/train.py`**:
    1.  **Data Loading (Spark):** Reads raw claims.
    2.  **Label Generation:** Applies rigid business rules to create "Ground Truth" labels (e.g., if `no_pre_auth` & `high_cost` -> `PEND`).
    3.  **Training:** Trains LightGBM to learn these rules and generalize to unseen patterns.
    4.  **Metric:** Optimizes for `multi_logloss` but validates on `precision`.

## 5. Edge Cases
*   **Ambiguity:** If features don't strongly indicate Approve/Deny, the model learns to predict `PEND`.
*   **Drift:** If policy rules change, the "Ground Truth" generation logic must be updated and the model retrained.

## 6. Future Improvements
*   **NLP:** Use claim notes/medical reports to reduce `PEND` rate for complex cases.
