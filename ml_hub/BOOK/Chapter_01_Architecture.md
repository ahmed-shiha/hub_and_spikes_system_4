# The MedIns Machine Learning Handbook

## Preface
This book serves as the definitive guide to the MedIns Machine Learning System. It is written for data engineers, data scientists, and ML engineers who need to understand not just *what* the system does, but *exactly how* it does it, down to the line of code.

## Table of Contents
1.  **Chapter 1: The Architecture of Scale** - Why we built it this way.
2.  **Chapter 2: The Core Hub (`medins_ml_utils`)** - The shared brain.
3.  **Chapter 3: The First Gate - Financial Approval** - Rejecting invalid claims early.
4.  **Chapter 4: The Core Business - Actuarial Pricing** - Predicting cost.
5.  **Chapter 5: The Shield - FWA Detection** - Finding fraud.
6.  **Chapter 6: Operations - STP & Network Value** - Automating the rest.

---

# Chapter 1: The Architecture of Scale

## 1.1 The Business Problem
Health insurance is complex. A single "claim" is a request for payment that triggers a cascade of questions:
*   Is the patient covered?
*   Is the procedure medically necessary?
*   Is the price fair?
*   Is the doctor a fraudster?
*   How much will this patient cost us next year?

In the past, these questions were answered by humans or rigid SQL rules. This does not scale to millions of members. We need Machine Learning.

## 1.2 The "Hub & Spoke" Solution
We faced a problem: if we build 5 different models, we might write the code to "calculate patient age" 5 times. Worse, one team might calculate it as `2023 - birth_year` and another as `(today - birth_date) / 365`. This creates **Feature Skew**.

To solve this, we built a **Hub & Spoke Architecture**:

### The Hub (`medins_ml_utils`)
This is a Python package that lives in `ml_hub/medins_ml_utils`. It is the *source of truth*.
*   **Data Connectors:** "Read the Claims Table" (Everyone reads it the same way).
*   **Features:** "Calculate Chronic Condition Score" (Everyone uses the exact same math).
*   **Utils:** "Load Config", "Setup Logs" (Standardized boilerplate).

### The Spokes (The Models)
These are independent projects (folders like `actuarial_pricing_model_v5`).
*   They **import** the Hub.
*   They define their own **Pipeline** (steps).
*   They own their own **Deployment** (Dockerfile).

This allows the Pricing Team to iterate fast without breaking the Fraud Team's code, while still sharing the core logic.

## 1.3 The Data Flow
1.  **Raw Data:** Sits in a Data Warehouse (simulated in `ml_hub/data/`).
2.  **Step 1 (Discovery):** scan data, learn global stats (e.g., "Average Cost of Diabetes"). Save as **Artifacts**.
3.  **Step 2 (ABT):** Use Spark to join massive tables into a flat "Analytical Base Table".
4.  **Step 3 (Feature Engineering):** Apply the **Hub's** logic to the ABT.
5.  **Step 4 (Training):** Train a model (LightGBM) on the features.
6.  **Step 5 (Serving):** Wrap the model in a REST API.

In the next chapter, we will open the hood of the **Hub** and see exactly how this shared logic works.
