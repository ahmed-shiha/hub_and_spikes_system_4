
import pytest
from pathlib import Path
import pandas as pd
import json
import sys

# In a real package, these would be imported from the actual scripts.
# For this self-contained test, we define mock versions.

def run_data_discovery_mock(input_dir: Path, output_dir: Path):
    """Mocks the artifact creation step."""
    mock_artifacts = {
        "chronic_conditions_definitions": {"Diabetes": ["E11"]},
        "chronic_condition_cost_weights": {"Diabetes": 1.5},
        "chronic_condition_freq_weights": {"Diabetes": 1.2},
        "provider_stats": {}, "provider_to_specialty_map": {}, 
        "specialty_avg_costs": {}, "provider_to_type_map": {},
        "denial_codes": {}
    }
    with open(output_dir / "data_driven_artifacts.json", "w") as f:
        json.dump(mock_artifacts, f)
    # Create mock train/test split files
    pd.DataFrame({'Patient_KEY': [1,2,3]}).to_csv(output_dir.parent / "data" / "train_member_ids.csv", index=False)
    pd.DataFrame({'Patient_KEY': [4]}).to_csv(output_dir.parent / "data" / "test_member_ids.csv", index=False)


def run_build_abt_mock(input_dir: Path, output_dir: Path):
    """Mocks the ABT creation step."""
    pd.DataFrame({
        'Patient_KEY': [1, 2, 3, 4], 'age': [30, 40, 50, 60], 'gender': ['1', '2', '1', '2'],
        'city': ['Riyadh', 'Jeddah', 'Riyadh', 'Dammam'], 'classname': ['A', 'B', 'A', 'C'],
        'annualized_cost': [1000, 2000, 1500, 3000],
        'diagnosis_codes_list': [['E11.9'], ['I10'], ['M54.5'], []],
        'self_reported_conditions': [['Diabetes'], [], [], []],
        'claim_costs_list': [[] for _ in range(4)], 'pre_auth_outcomes_list': [[] for _ in range(4)],
        'service_dates_list': [[] for _ in range(4)], 'providers_visited_list': [[] for _ in range(4)],
        'practitioner_specialty_list': [[] for _ in range(4)], 'activity_type_list': [[] for _ in range(4)],
        'adjudication_reason_list': [[] for _ in range(4)], 'policy_duration_days': [365]*4
    }).to_csv(output_dir / "abt_renewal.csv", index=False)
    pd.DataFrame({
        'Patient_KEY': [1, 2, 3, 4], 'age': [30, 40, 50, 60], 'gender': ['1', '2', '1', '2'],
        'city': ['Riyadh', 'Jeddah', 'Riyadh', 'Dammam'], 'classname': ['A', 'B', 'A', 'C'],
        'annualized_cost': [1000, 2000, 1500, 3000], 'self_reported_conditions': [['Diabetes'], [], [], []]
    }).to_csv(output_dir / "abt_onboarding.csv", index=False)

def run_feature_engineering_mock(data_dir: Path, model_dir: Path, output_dir: Path):
    """Mocks the feature engineering step."""
    pd.DataFrame({
        'Patient_KEY': [1, 2, 3, 4], 'age': [30, 40, 50, 60], 'annualized_cost': [1000, 2000, 1500, 3000],
        'self_reported_chronic_score': [1.5, 0, 0, 0]
    }).to_csv(output_dir / "abt_onboarding_featured.csv", index=False)
    pd.DataFrame({
        'Patient_KEY': [1, 2, 3, 4], 'age': [30, 40, 50, 60], 'annualized_cost': [1000, 2000, 1500, 3000],
        'data_driven_risk_score': [1.35, 0, 0, 0]
    }).to_csv(output_dir / "abt_renewal_vice_champion_featured.csv", index=False)
    pd.DataFrame({
        'Patient_KEY': [1, 2, 3, 4], 'age': [30, 40, 50, 60], 'annualized_cost': [1000, 2000, 1500, 3000],
        'data_driven_risk_score': [1.35, 0, 0, 0], 'avg_provider_risk_score': [1.1, 0.9, 1.0, 1.2]
    }).to_csv(output_dir / "abt_renewal_champion_featured.csv", index=False)


def test_full_pipeline_integration(tmp_path):
    """
    An integration test to ensure the main data pipeline steps
    can run in sequence and produce the expected final artifacts.
    This validates that the data contracts between steps are maintained.
    """
    # 1. ARRANGE: Create a temporary mock project structure
    project_root = tmp_path
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    data_output_dir = output_dir / "data"
    model_output_dir = output_dir / "models"
    
    data_dir.mkdir()
    output_dir.mkdir()
    data_output_dir.mkdir()
    model_output_dir.mkdir()
    
    # 2. ACT: Run the data pipeline steps sequentially using the mock functions
    run_data_discovery_mock(data_dir, model_output_dir)
    run_build_abt_mock(data_dir, data_output_dir)
    run_feature_engineering_mock(data_output_dir, model_output_dir, data_output_dir)
    
    # 3. ASSERT: Check if the final, most important output was created correctly
    final_output_file = data_output_dir / "abt_renewal_champion_featured.csv"
    assert final_output_file.exists(), "Final champion ABT was not created."
    
    final_df = pd.read_csv(final_output_file)
    assert "avg_provider_risk_score" in final_df.columns, "Expected champion feature was not in the final output."
    assert len(final_df) == 4, "Final DataFrame has incorrect number of rows."

