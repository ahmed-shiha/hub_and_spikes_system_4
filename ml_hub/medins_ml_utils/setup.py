from setuptools import setup, find_packages

setup(
    name="medins_ml_utils",
    version="0.2.0",
    packages=find_packages(),
    description="Shared utilities, data connectors, and feature logic for MedIns ML projects.",
    author="ML Platform Team",
    install_requires=[
        "pandas", "numpy", "scikit-learn", "lightgbm", "scipy", "mlflow", "pyspark"
    ],
    extras_require={
        "dev": ["pytest"],
    }
)
