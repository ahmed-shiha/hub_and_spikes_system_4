# STP Automation Model v1: Comprehensive Documentation

This document provides a detailed overview of the Straight-Through Processing (STP) Automation Model, a machine learning system designed to automate the adjudication of simple health insurance claims.

---

### 1. Primary Business Problem & Model Objective

**Business Problem:** A large volume of health insurance claims are simple and unambiguous. They are for common, covered services, have prior authorization, and are well within policy limits. Forcing these claims through a manual review process creates a significant operational bottleneck, leading to:
-   **High Adjudication Costs:** Manual processing of every claim is expensive and does not scale.
-   **Slow Payment Cycles:** Delays in approving simple claims can frustrate providers and members.
-   **Inefficient Use of Skilled Adjudicators:** Expert human adjudicators spend too much of their time on "rubber-stamp" approvals instead of focusing on complex, high-risk cases.

**Model Objective:** The objective of this model is to **automate the adjudication of low-complexity claims with extremely high precision**. The model is a multi-class classifier that assigns each claim to one of three categories:
-   `APPROVE`: The claim is clearly valid and should be paid automatically.
-   `DENY`: The claim is clearly invalid (e.g., for a non-covered service) and should be rejected automatically.
-   `PEND`: The claim is ambiguous, complex, or borderline, and requires human review.

The primary goal is to maximize the number of claims that can be safely classified as `APPROVE` or `DENY`, thereby increasing the Straight-Through Processing (STP) rate.

---

### 2. Scope & Boundaries

**In-Scope:**
-   **Automation of Simple Claims:** The model is specifically designed to handle high-volume, low-complexity claims.
-   **Multi-Class Classification:** It predicts one of three discrete outcomes, serving as a triage system for the adjudication process.
-   **Emphasis on Precision:** The model is heavily biased towards precision. If it is not highly confident in an `APPROVE` or `DENY` prediction, it is designed to default to `PEND`. The business requirement is >99% precision for automated decisions.

**Out-of-Scope:**
-   **Complex Case Adjudication:** The model is not intended to make decisions on complex medical cases, high-cost claims, or claims with unusual characteristics. Its job is to *identify* these cases and send them to a human.
-   **FWA or Pricing:** This model does not perform fraud detection or cost prediction. It is solely focused on the adjudication decision based on the information presented in the claim.

---

### 3. Key Performance Indicators (KPIs) & Success Metrics

Success is measured by the model's ability to increase automation without introducing errors.

**Business KPIs:**
-   **Straight-Through Processing (STP) Rate:** The percentage of total claims that are automatically adjudicated (`APPROVE` or `DENY`) without human intervention.
-   **Adjudication Cost per Claim:** Should decrease as the STP rate increases.
-   **Adjudication Turnaround Time:** The average time from claim submission to decision should decrease.

**Technical Metrics:**
-   **Precision (for `APPROVE` and `DENY` classes):** This is the most critical metric. Of all the claims the model auto-approved, what percentage were correctly approved? Of all the claims it auto-denied, what percentage were correctly denied? The target for this is >99%.
-   **`PEND` Rate:** The percentage of claims the model sends for manual review. While a high STP rate is good, the `PEND` rate must be sufficient to catch the truly complex cases.
-   **Multi-class Logloss:** The loss function used during training to optimize the model's predicted probabilities.

---

### 4. Data Dictionary & Lineage

The model is trained on a "one-row-per-claim" ABT.

| Column Name             | Data Type     | Source System             | Description                                                                                              |
| ----------------------- | ------------- | ------------------------- | -------------------------------------------------------------------------------------------------------- |
| `claim_id`              | `String`      | Claims System             | Unique identifier for the claim.                                                                         |
| `net_amount`            | `Double`      | Claims System             | The net amount to be paid after deductibles.                                                             |
| `has_pre_auth`          | `Boolean`     | Pre-Authorization System  | `True` if the service had prior approval.                                                                |
| `service_is_covered`    | `Boolean`     | Policy System             | `True` if the service code is listed as a covered benefit under the member's plan.                         |
| `is_complex_diag`       | `Boolean`     | Medical Codes Database    | `True` if the primary diagnosis code is associated with a complex or chronic condition.                    |
| `adjudication_decision` | `String`      | Adjudication System       | **Target Variable:** The final decision (`APPROVE`, `DENY`, `PEND`) generated by a rules-based "teacher" process. |

**Data Lineage:**
1.  **Source Systems (Simulated):** Raw data from claims, policy, and pre-auth systems.
2.  **Label Generation (`train.py`):** A critical step where a set of **rigid business rules** is applied to the raw data to generate the "ground truth" `adjudication_decision` labels. For example:
    -   `IF has_pre_auth AND service_is_covered AND net_amount < 500 THEN 'APPROVE'`
    -   `IF NOT service_is_covered THEN 'DENY'`
    -   `IF is_complex_diag OR net_amount > 5000 THEN 'PEND'`
3.  **Training:** The LightGBM model is trained to **learn and generalize these rules**, effectively creating a "soft" version of the rule book that can handle more nuanced cases.

---

### 5. Feature Engineering & Logic

The features for this model are relatively simple and are often binary flags derived from upstream systems.

-   **`has_pre_auth`:**
    -   **Purpose:** One of the strongest signals for approval. If a service was pre-authorized, it is highly likely to be paid.
-   **`service_is_covered`:**
    -   **Purpose:** A direct check against the member's policy benefits. A fundamental requirement for payment.
-   **`is_complex_diag`:**
    -   **Purpose:** A flag used to identify claims that should not be automated. Complex medical cases require human expertise to adjudicate correctly.
-   **`net_amount`:**
    -   **Purpose:** High-value claims represent a greater financial risk and are therefore always sent for human review.

---

### 6. Model vs. Rules-Based Logic

This model represents a "student/teacher" paradigm.

-   **Rules-Based Logic (The "Teacher"):** The ground truth labels are generated by a deterministic, hard-coded set of business rules. This rule set is the "perfect" adjudicator.
-   **Machine Learning Model (The "Student"):** The LightGBM model's job is to **learn the patterns** created by these rules. The reason for this is twofold:
    1.  **Generalization:** The ML model can sometimes learn to handle combinations of features that are not explicitly coded in the rules, making it more robust.
    2.  **Efficiency:** Executing a single ML model can be much more computationally efficient at scale than running a complex, multi-stage rule engine for every claim.

The model is essentially a **surrogate** for the complex rule engine, optimized for high-speed, high-precision execution.

---

### 7. Stakeholders & Downstream Consumers

-   **Claims Adjudication Team:** The primary stakeholders. The model automates their simplest tasks, freeing them up to focus on high-value work.
-   **Business Process Automation Team:** Responsible for the overall STP rate and the efficiency of the claims pipeline.
-   **Adjudication System:** The downstream system that either processes the `APPROVE`/`DENY` decision or routes the `PEND` cases to a human work queue.

---

### 8. Risk Assessment & Mitigation

-   **Risk of Incorrect Automation (Precision Failure):**
    -   **Risk:** The model could auto-approve a claim that should be denied (overpayment) or auto-deny one that should be approved (provider/member abrasion). This is the single biggest risk.
    -   **Mitigation:** The model is evaluated and tuned using a probability threshold that maximizes precision to a level of 99% or higher. Any claim where the model's confidence score is below this high threshold is automatically classified as `PEND`.

-   **Risk of Rule Changes (Concept Drift):**
    -   **Risk:** If the business rules for adjudication change (e.g., a new service is now covered), the model's "teacher" will be giving it outdated information, and its predictions will become incorrect.
    -   **Mitigation:** The model must be retrained whenever the upstream "teacher" logic for label generation is updated. This requires a strong MLOps process that links the model's training pipeline to the business rule repository.

---

### 9. Scenario Walkthroughs & Examples

**Scenario 1: Simple Approval**
-   **Claim Data:**
    -   `net_amount`: $250
    -   `has_pre_auth`: True
    -   `service_is_covered`: True
    -   `is_complex_diag`: False
-   **Model Prediction:** The model predicts `APPROVE` with very high confidence (e.g., 99.8%).
-   **Outcome:** The claim is auto-approved and paid.

**Scenario 2: Simple Denial**
-   **Claim Data:**
    -   `net_amount`: $100
    -   `has_pre_auth`: False
    -   `service_is_covered`: False
-   **Model Prediction:** The model predicts `DENY` with very high confidence (e.g., 99.9%).
-   **Outcome:** The claim is auto-denied.

**Scenario 3: Ambiguous Case**
-   **Claim Data:**
    -   `net_amount`: $800
    -   `has_pre_auth`: True
    -   `service_is_covered`: True
    -   `is_complex_diag`: True
-   **Model Prediction:** Although the claim has pre-auth and is for a covered service, the presence of a complex diagnosis makes the model unsure. It cannot confidently predict `APPROVE`.
-   **Outcome:** The model predicts `PEND`, and the claim is routed to a human adjudicator for review.

---

### 10. Operational & Implementation Details

-   **Deployment:** The model is typically deployed as a real-time service that the main claims processing pipeline calls for each new claim.
-   **Retraining:** The model must be retrained immediately following any update to the business rules used for label generation.
-   **Monitoring:** It is critical to monitor the model's precision in production. A small, random sample of auto-adjudicated claims should be audited by humans to ensure the model's real-world precision matches its performance in testing.
