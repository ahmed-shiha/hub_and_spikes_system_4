
import pytest
from pyspark.sql import SparkSession
from ml_hub.medins_ml_utils.medins_ml_utils.features import create_onboarding_features

@pytest.fixture(scope="session")
def spark_session():
    """Creates a SparkSession for use in tests."""
    return SparkSession.builder.appName("test_features").getOrCreate()

def test_create_onboarding_features(spark_session):
    """Tests the create_onboarding_features function."""
    # Create a dummy DataFrame
    df = spark_session.createDataFrame(
        [
            (1, "urban", 25, ["Diabetes", "Hypertension"]),
            (2, "suburban", 45, ["Heart Disease"]),
            (3, "rural", 65, []),
        ],
        ["patient_id", "location", "age", "self_reported_conditions"],
    )

    # Create dummy artifacts
    artifacts = {
        "chronic_condition_cost_weights": {
            "Diabetes": 1.5,
            "Hypertension": 1.2,
            "Heart Disease": 2.0,
        },
        "chronic_condition_freq_weights": {
            "Diabetes": 1.1,
            "Hypertension": 1.3,
            "Heart Disease": 1.8,
        },
    }

    # Apply the feature engineering function
    df_out = create_onboarding_features(df, artifacts)

    # Check that the new columns have been added
    assert "self_reported_chronic_score" in df_out.columns
    assert "age_x_chronic_score" in df_out.columns
    assert "demographic_risk_score" in df_out.columns

    # Check that the values in the new columns are correct
    expected_values = [
        (1, 2.55, 63.75, 0.3),
        (2, 1.9, 85.5, 0.45),
        (3, 0.0, 0.0, 0.585),
    ]
    actual_values = [
        (
            row.patient_id,
            round(row.self_reported_chronic_score, 2),
            round(row.age_x_chronic_score, 2),
            round(row.demographic_risk_score, 3),
        )
        for row in df_out.select(
            "patient_id",
            "self_reported_chronic_score",
            "age_x_chronic_score",
            "demographic_risk_score",
        ).collect()
    ]
    assert actual_values == expected_values
