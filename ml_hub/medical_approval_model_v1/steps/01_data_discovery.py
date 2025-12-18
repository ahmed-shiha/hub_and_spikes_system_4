
from medins_ml_utils.utils import load_config, setup_directories, save_artifacts, get_spark_session
from medins_ml_utils.data_connectors import get_data
import logging

def run_step():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    setup_directories(config)
    spark = get_spark_session("MedicalApproval_Discovery")
    
    # Discovery: Learn "Standard of Care"
    # Map Diagnosis -> Allowed Procedures (Simulated)
    # In reality, we'd mine F_Claim_Item
    
    artifacts = {
        "standard_of_care": {
            "E11": ["PROC_INSULIN", "PROC_CHECKUP", "PROC_LAB"], # Diabetes
            "I10": ["PROC_EKG", "PROC_CHECKUP"], # Hypertension
            "J00": ["PROC_CHECKUP"] # Cold
        },
        "gender_restricted_procedures": {
            "PROC_C_SECTION": ["F"],
            "PROC_PROSTATE_EXAM": ["M"]
        }
    }
    
    save_artifacts(artifacts, config['paths']['output_dir'] + "/models")

if __name__ == "__main__":
    run_step()
