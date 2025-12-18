
from medins_ml_utils.utils import load_config, setup_directories, save_artifacts, get_spark_session
from medins_ml_utils.data_connectors import get_data
import logging

def run_step():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    setup_directories(config)
    spark = get_spark_session("FinancialApproval_Discovery")
    
    # Load Data
    data = get_data(config['paths']['data_dir'], ["F_Claim_Header"], spark)
    df = data["F_Claim_Header"]
    
    # Discovery: Calculate Plan Limits (Simulated)
    # in reality, this comes from a Plan Master table.
    artifacts = {
        "plan_limits": {
            "P001": 5000.0,
            "P002": 10000.0,
            "P003": 1000000.0
        },
        "provider_master": {
            "PROV1": {"tier": 1},
            "PROV2": {"tier": 2},
            "PROV3": {"tier": 1},
            "PROV4": {"tier": 3} # Blacklisted
        }
    }
    
    save_artifacts(artifacts, config['paths']['output_dir'] + "/models")

if __name__ == "__main__":
    run_step()
