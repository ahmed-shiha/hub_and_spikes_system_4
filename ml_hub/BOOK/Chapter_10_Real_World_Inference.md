# Chapter 10: Real World Inference - Handling Messy Data

In the lab, we train on clean, star-schema CSVs (`F_Claim_Header`, `D_Patient`).
In the real world, we receive messy dumps like `TB Medical Claim Request.csv`.

## 10.1 The "Tuba" Data Format
The provided CSV is a **denormalized report**.
*   **One Row per Claim?** No. It seems to have multiple sections columns (`(Service Items)`, `(Medications)`).
*   **Hierarchy:**
    *   Claim Header (`CLM-2025...`)
    *   --> User Info (`EMP-443`)
    *   --> Service Items (List)
    *   --> Medications (List)

## 10.2 The User JSON Object
We receive user context as a JSON blob.
```json
{
  "salary": "<4000 SAR",
  "had_chronic_diseases": 0,
  "gender": "Male",
  "birth_date": "1998-01-01",
  "marital_status": "Single"
}
```
**Feature Opportunities:**
*   **`salary`**: Financial Risk. Low salary might correlate with higher fraud risk (reselling meds) or lower health literacy? (Ethical risk here, be careful).
*   **`had_chronic_diseases`**: Medical Necessity. If `0`, but claiming Diabetes meds -> Suspicious.
*   **`marital_status`**: Policy Validation. If "Single" but claiming "Maternity" -> Check rules.

## 10.3 The Inference Challenge
We cannot simply `spark.read.csv`. We must:
1.  **Parse:** Handle the column names with parentheses.
2.  **Explode:** The CSV might imply one-to-many relationships if columns are repeated or comma-separated? (Actually, standard CSV usually duplicates the header for each item, or uses wide format).
    *   *Looking at the head:* It seems to be a "Wide" format where sections are columns? Or a join export?
    *   *Correction:* It looks like a standard export where `ID (Service Items)` defines the grain.

## 10.4 The Mapping Strategy

| Real World Field | Model Feature | Logic |
| :--- | :--- | :--- |
| `Gender` (User JSON) | `Gender` | Direct map. |
| `Diagnosis Code (Service Items)` | `diagnosis_code` | Direct map. |
| `Service Code (Service Items)` | `procedure_code` | Direct map. |
| `Billed Net Amount` | `claim_amount` | Direct map. |
| `had_chronic_diseases` (User JSON) | `self_reported_chronic_score` | If 1 -> High Score. |

## 10.5 Missing Labels
In inference, `Is Approved?` columns are either empty or "Draft".
**The Goal:** Our model must fill these columns.
*   `Financial Model` -> fills `Status (Financial)`
*   `Medical Model` -> fills `Status (Medical)`
