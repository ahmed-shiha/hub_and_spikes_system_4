# Reference: Financial Claim Approval Model (v1)

## 1. Business Context
**The Problem:** Many claims are rejected not because of medical necessity, but due to administrative or financial reasons (e.g., plan limits reached, coverage terminated, out-of-network provider).
**The Solution:** A model to predict "Financial Rejection" probability before clinical review. This reduces the load on clinical adjudicators (STP) and allows for faster feedback to providers.

## 2. Model Methodology
We use a **Binary Classifier** (LightGBM).
*   **Target:** `is_financially_rejected` (1 = Reject, 0 = Pass).
*   **Features:** Plan utilization, network status, coordination of benefits, historical billing errors.

## 3. Detailed Feature Reference

### 3.1 `plan_utilization_percent`
*   **Logic:** (Current Claim + YTD Spend) / Annual Plan Limit.
*   **Insight:** As members near their limit, the probability of partial or full rejection increases.

### 3.2 `is_out_of_network`
*   **Logic:** Checks if the provider is contracted with the member's specific plan.
*   **Insight:** Out-of-network claims often require different adjudication rules or are rejected outright depending on the plan type (e.g., HMO vs PPO).

### 3.3 `cob_flag`
*   **Logic:** "Coordination of Benefits". Indicates if the member has other insurance.
*   **Insight:** If another payer is primary, we might reject until they pay.

## 4. Technical Implementation
Runs as a pre-check before STP.
