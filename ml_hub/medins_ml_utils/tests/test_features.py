
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from pyspark.sql import SparkSession

# Add project path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'medins_ml_utils'))
from features import create_renewal_vice_champion_features

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .appName("TestApp") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

@pytest.fixture
def mock_artifacts():
    return {
        'chronic_condition_cost_weights': {'Diabetes': 1.8, 'Hypertension': 1.2},
        'chronic_condition_freq_weights': {'Diabetes': 1.2, 'Hypertension': 1.0},
        'chronic_conditions_definitions': {'Diabetes': ['E11'], 'Hypertension': ['I10']},
        'denial_codes': {}
    }

def test_data_driven_risk_score(spark, mock_artifacts):
    # 1. ARRANGE: Create mock input data using Spark DataFrame
    # Note: Spark handles arrays and nulls differently. 
    
    data = [
        (['E11.9', 'I10'], [], [], [], [], [], [], '', '', ''),
        (['E11.0'], [], [], [], [], [], [], '', '', ''),
        (['R51'], [], [], [], [], [], [], '', '', ''),
        ([], [], [], [], [], [], [], '', '', ''),
        (None, [], [], [], [], [], [], '', '', ''), 
        # Skip 'not_a_list' as Spark schema enforces types. Schema mismatch would be an ETL error, not feature logic error.
    ]
    
    schema = ["diagnosis_codes_list", "claim_costs_list", "pre_auth_outcomes_list", "service_dates_list",
              "practitioner_specialty_list", "activity_type_list", "adjudication_reason_list", 
              "gender", "city", "classname"]
    
    # Needs complex schema for arrays
    # Simplified for test:
    # We let Spark infer, but for empty lists it might be ambiguous.
    
    # Let's use createDataFrame with pandas to be safe on types or force schema
    pdf = pd.DataFrame(data, columns=schema)
    mock_df = spark.createDataFrame(pdf)
    
    # 2. ACT
    result_df = create_renewal_vice_champion_features(mock_df, mock_artifacts)
    
    # 3. ASSERT
    # Collect to local to check
    results = result_df.select("data_driven_risk_score").collect()
    scores = [r.data_driven_risk_score for r in results]
    
    expected_scores = [
        1.5 + 1.1,  # Diabetes + Hypertension
        1.5,        # Just Diabetes
        0.0,        # No chronic conditions
        0.0,        # Empty list
        0.0,        # Null input
    ]
    
    np.testing.assert_allclose(scores, expected_scores, atol=0.01)
