# Reference: STP Automation Model (v1)

## 1. Business Context
**The Problem:** Manually reviewing a medical claim costs MedIns ~$10 - $20 per claim. With millions of claims, this is unsustainable.
**The Solution:** Straight-Through Processing (STP). An AI "Adjudicator" that handles the easy cases.

**The "Pend" Philosophy:**
*   An AI mistake is costly.
    *   False Approve: We lose money paying a bad claim.
    *   False Deny: We anger a customer and face regulatory fines.
*   **Strategy:** If the AI is not 99% sure, it should predict **`PEND`**. This sends the claim to a human. We optimize for "Automation Rate" only *after* securing "Precision".

## 2. Model Methodology
We use a **Multi-Class Classifier** (LightGBM).
*   **Classes:** `APPROVE`, `DENY`, `PEND`.
*   **Objective:** `multi_logloss`.

## 3. Detailed Feature Reference

### 3.1 `has_pre_auth`
*   **Logic:** Binary (1 if Pre-Authorization Reference exists).
*   **Insight:** If a doctor already got approval for a surgery, the claim is 99% safe to pay. This is the strongest signal for `APPROVE`.

### 3.2 `is_complex_diag`
*   **Logic:** Flag based on ICD-10 chapters (e.g., Oncology/Cancer is complex).
*   **Insight:** Complex diseases have complex billing rules. The model learns to `PEND` these for human specialist review.

### 3.3 `net_amount`
*   **Logic:** Total claim value.
*   **Insight:** Low value claims (< $50) are low risk. It's cheaper to auto-pay than to pay a human $20 to review it. The model learns a "Safe Threshold" dynamically.

## 4. Technical Implementation

### Rules vs Model
Why use ML? Why not just `if amount < 50 then pay`?
*   **Complexity:** Rules explode (if < 50 AND not Cancer AND provider is Trusted...).
*   **Maintenance:** ML learns the interactions. If "Trusted Provider" turns bad (FWA score rises), the ML naturally stops auto-approving them without a code change.

### Validation Gate
The `config.json` enforces strict safety:
```json
"validation_thresholds": {
    "min_approve_precision": 0.95,
    "min_deny_precision": 0.95
}
```
If the trained model achieves 94% precision, the CI/CD pipeline **fails**. We do not deploy unsafe adjudicators.
