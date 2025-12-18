
import pandas as pd
from pathlib import Path
import logging
from pyspark.sql import SparkSession

def get_spark_session(app_name="MedInsMLHub"):
    """Creates or gets a SparkSession."""
    return SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()

def get_data(data_dir: Path, tables_to_load: list, spark: SparkSession = None) -> dict:
    """
    The single, governed entry point for accessing data in the ML Hub.

    IN PRODUCTION: This function would use Spark connectors (Snowflake/BigQuery/Delta Lake).
    
    FOR SIMULATION: Reads from local CSVs in the central `ml_hub/data/` directory 
    using PySpark.
    """
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    if spark is None:
        spark = get_spark_session()

    logging.info(f"SIMULATION: Loading {len(tables_to_load)} tables from central data directory: {data_dir.resolve()} using Spark...")
    try:
        data = {}
        for name in tables_to_load:
            # Handle potential variations in filenames (e.g., d_ vs D_)
            possible_filenames = [f"{name}.csv", f"d_{name}.csv", f"D_{name}.csv", f"f_{name}.csv", f"F_{name}.csv"]
            found = False
            for filename in possible_filenames:
                filepath = data_dir / filename
                if filepath.exists():
                    # Read CSV with Spark
                    # inferSchema=True creates overhead but is useful for simulation. 
                    # In prod, schema should be defined.
                    df = spark.read.option("header", "true") \
                                   .option("inferSchema", "true") \
                                   .csv(str(filepath))
                    data[name] = df
                    found = True
                    break
            if not found:
                 raise FileNotFoundError(f"Could not find a valid CSV for table '{name}' in {data_dir}")
        return data
    except FileNotFoundError as e:
        logging.error(f"A required data file was not found: {e}. Please check the 'ml_hub/data' directory.")
        raise
