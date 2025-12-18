# Chapter 8: The Full Picture - System Analysis and Future State

This chapter provides a comprehensive, "maliciously" fine-grained analysis of the current data landscape and proposes a more mature architecture for Claims Adjudication.

## 8.1 Data Ecosystem Analysis

We have analyzed the following tables in `ml_hub/data/`:

| Table | Content | Analysis & Limitations |
| :--- | :--- | :--- |
| **`F_Claim_Header`** | The spine. `Patient_KEY`, `Provider`, `Amount`, `Date`. | **Problem:** Lacks `Diagnosis_Code` (it's in Item). Lacks `Procedure_Code` (crucial for medical necessity). **Opportunity:** Has `max_serviced_date`, good for velocity checks (FWA). |
| **`F_Claim_Item`** | Line items. `Diagnosis_Code`. | **Problem:** One header can have many items. We must aggregate (set of diagnoses). **Constraint:** Currently missing `Procedure_Code` (CPT/HCPCS). We cannot check "Does Diagnosis match Procedure?" without this. |
| **`D_Provider`** | `Specialty`. | **Crucial:** Used for "Peer Grouping". A Cardiologist billing for an MRI is rare; a Radiologist is normal. We must fingerprint by Specialty. |
| **`D_Patient`** | `Age`, `Gender`. | **Demographics:** Essential for "Medical Necessity" (e.g., Maternity claims for Males = Deny). |

## 8.2 The "Approval" Paradox

Currently, our **STP Model** tries to do everything:
*   "Is the price low?" (Financial)
*   "Is it covered?" (Contractual)
*   "Is it medically needed?" (Clinical)

**Critique:** This is a "God Model". It is hard to debug. If it denies a claim, we don't know if it's because the price was high or the diagnosis was wrong.

## 8.3 The Proposed "Assembly Line" Architecture

We should decompose "Claim Approval" into three distinct, specialized models:

1.  **Gate 1: Financial Approval (Implemented)**
    *   **Question:** "Is this bill administratively valid?"
    *   **Features:** Plan Limits, Network Status (Tier), Duplicate Check.
    *   **Action:** If Fail -> **REJECT** (Administrative Denial).

2.  **Gate 2: Medical Approval (New Proposal)**
    *   **Question:** "Is this treatment appropriate for this patient?"
    *   **Features:**
        *   `diag_proc_consistency`: Does diagnosis E11 (Diabetes) match procedure "Insulin Therapy"?
        *   `age_gender_consistency`: Does Gender 'M' match procedure "C-Section"?
        *   `frequency_check`: 5 MRIs in 1 week?
    *   **Action:** If Fail -> **DENY** (Medical Necessity).

3.  **Gate 3: FWA Shield (Implemented)**
    *   **Question:** "Is this provider lying?"
    *   **Features:** Weekend billing, Upcoding patterns.
    *   **Action:** If Fail -> **FLAG** (SIU Review).

4.  **Gate 4: STP Operations (The Coordinator)**
    *   **Question:** "Given Gates 1, 2, and 3 passed, can we skip the human?"
    *   **Logic:** If `Financial_Prob > 0.99` AND `Medical_Prob > 0.99` AND `FWA_Risk < 0.01` -> **AUTO-PAY**.

## 8.4 New Feature Engineering Requirements

To support the **Medical Approval Model**, we need:
*   **EDA Step:** We need to statistically map `Diagnosis` <-> `Procedure`. (e.g., Top 5 procedures for "Diabetes").
*   **Consistency Features:** Distance from the "Standard of Care".

## 8.5 The "Cold Start" Advantage
Even with a new insurer, biological facts do not change.
*   "Men do not give birth" is a universal rule (Artifact).
*   "Appendicitis requires Appendectomy" is a universal pattern.
*   **Strategy:** We pre-train the **Medical Approval Model** on the General Data Repository. It learns "Standard of Care". We ship this model to Day 1.

This concludes the architectural analysis. The system is moving from "Monolithic" to "Modular & Specialized".
