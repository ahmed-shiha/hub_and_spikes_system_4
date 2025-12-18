# How to Run MedIns ML System

This guide explains how to run the Machine Learning pipelines in two environments:
1.  **Local PC** (Windows/Mac/Linux)
2.  **Google Colab** (Cloud)

---

## Option 1: Running on Local PC

### Prerequisites
1.  **Python 3.8+** installed.
2.  **Java (JDK 8, 11, or 17)**: PySpark requires Java.
    *   *Check:* Open terminal/cmd and run `java -version`.
    *   *Install:* If missing, download OpenJDK or Oracle JDK.

### Step-by-Step Instructions

#### 1. Setup the Environment
Open your terminal or command prompt and navigate to the project folder (where `ml_hub` is located).

```bash
# 1. Install the core shared library (The Hub) in editable mode
pip install -e ml_hub/medins_ml_utils

# 2. Install external dependencies
pip install pyspark lightgbm mlflow pandas scikit-learn
```

#### 2. Run the Actuarial Pricing Model
This is the main model. Run the steps in order:

```bash
# Step 1: Data Discovery (Profiles data & calculates stats)
python ml_hub/actuarial_pricing_model_v5/steps/01_data_discovery.py

# Step 2: Build Analytical Base Table (Spark ETL - Heavy Lifting)
python ml_hub/actuarial_pricing_model_v5/steps/02_build_abt.py

# Step 3: Feature Engineering (Applies standardized logic)
python ml_hub/actuarial_pricing_model_v5/steps/03_feature_engineering.py

# Step 4: Feature Selection (Correlation Analysis)
python ml_hub/actuarial_pricing_model_v5/steps/04_feature_selection.py

# Step 5: Training (LightGBM + MLflow logging)
python ml_hub/actuarial_pricing_model_v5/steps/05_gradient_boosting.py
```

#### 3. Check Results
*   **Data Outputs:** Check `ml_hub/actuarial_pricing_model_v5/steps/output/data/` for `.parquet` and `.csv` files.
*   **Model Artifacts:** Check `mlruns/` folder or `ml_hub/actuarial_pricing_model_v5/steps/output/models/`.

---

## Option 2: Running on Google Colab

Google Colab is a free cloud environment. Since PySpark requires Java, we need a specific setup script.

### 1. Prepare the Zip File
1.  Zip the entire `ml_hub` folder (and this `HOW_TO_RUN.md` file if you want).
2.  Name it `project.zip`.

### 2. Upload to Colab
1.  Go to [colab.research.google.com](https://colab.research.google.com/).
2.  Create a **New Notebook**.
3.  On the left sidebar, click the **Folder (Files)** icon.
4.  Click the **Upload** button and select your `project.zip`.

### 3. Run the following Code Cells

**Cell 1: Unzip the Project**
```python
!unzip -q project.zip
print("Project unzipped!")
```

**Cell 2: Install Java (Required for PySpark)**
```python
# Install OpenJDK 8
!apt-get install openjdk-8-jdk-headless -qq > /dev/null

# Set JAVA_HOME environment variable
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
print("Java installed!")
```

**Cell 3: Install Python Dependencies**
```python
# Install PySpark and ML libraries
!pip install pyspark lightgbm mlflow

# Install our local 'Hub' library
!pip install -e ml_hub/medins_ml_utils
print("Dependencies installed!")
```

**Cell 4: Run the Pricing Pipeline**
```python
# Run Step 1: Data Discovery
!python ml_hub/actuarial_pricing_model_v5/steps/01_data_discovery.py

# Run Step 2: Build ABT
!python ml_hub/actuarial_pricing_model_v5/steps/02_build_abt.py

# Run Step 3: Feature Engineering
!python ml_hub/actuarial_pricing_model_v5/steps/03_feature_engineering.py

# Run Step 5: Training
!python ml_hub/actuarial_pricing_model_v5/steps/05_gradient_boosting.py
```

### 4. Download Results
1.  In the **Files** sidebar, navigate to `ml_hub/actuarial_pricing_model_v5/steps/output`.
2.  Right-click on files (like `y_pred_renewal_champion.csv`) and select **Download**.

---

## Troubleshooting

*   **"ModuleNotFoundError: No module named 'medins_ml_utils'":**
    *   Make sure you ran `pip install -e ml_hub/medins_ml_utils`.
    *   In Colab, ensure the path `ml_hub/medins_ml_utils` exists after unzipping.

*   **"Java not found":**
    *   Ensure you ran the Java installation cell (Cell 2) in Colab.
    *   On PC, ensure `JAVA_HOME` is set in your system environment variables.
