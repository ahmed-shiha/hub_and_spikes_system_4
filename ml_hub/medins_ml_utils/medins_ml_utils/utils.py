
import json
from pathlib import Path
import logging
from pyspark.sql import SparkSession

def get_spark_session(app_name="MedInsMLHub"):
    """
    Creates or retrieves a SparkSession.
    Configures it for local simulation if no master is set.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

def load_config(config_path: str = 'config.json') -> dict:
    """
    Loads the project configuration from a JSON file.
    Assumes the script is run from a project root where config.json is located.
    """
    try:
        full_config_path = Path.cwd() / config_path
        with open(full_config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {full_config_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the configuration file: {full_config_path}")
        raise

def setup_directories(config: dict) -> dict:
    """Creates all necessary output directories defined in the config."""
    paths = config['paths']
    project_root = Path.cwd()
    
    output_dir = project_root / paths['output_dir']
    data_output_dir = output_dir / "data"
    model_output_dir = output_dir / "models"
    results_output_dir = output_dir / "results"

    output_dir.mkdir(exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    results_output_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "data_dir": project_root / paths['data_dir'],
        "output_dir": output_dir,
        "data_output_dir": data_output_dir,
        "model_output_dir": model_output_dir,
        "results_output_dir": results_output_dir,
    }

def save_artifacts(artifacts: dict, model_output_dir: Path):
    """Saves the calculated artifacts dictionary to a JSON file."""
    if not isinstance(model_output_dir, Path):
        model_output_dir = Path(model_output_dir)
        
    artifacts_path = model_output_dir / 'data_driven_artifacts.json'
    try:
        with open(artifacts_path, 'w') as f:
            json.dump(artifacts, f, indent=4)
        logging.info(f"Data-driven artifacts saved to {artifacts_path}")
    except Exception as e:
        logging.error(f"Failed to save artifacts: {e}")
        raise

def load_artifacts(model_output_dir: Path) -> dict:
    """Loads the calculated artifacts dictionary from a JSON file."""
    artifacts_path = model_output_dir / 'data_driven_artifacts.json'
    try:
        with open(artifacts_path, 'r') as f:
            artifacts = json.load(f)
        logging.info(f"Successfully loaded data-driven artifacts from {artifacts_path}")
        return artifacts
    except FileNotFoundError:
        logging.error(f"Artifacts file not found at: {artifacts_path}. Make sure Stage 1 ran successfully.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the artifacts file: {artifacts_path}")
        raise
