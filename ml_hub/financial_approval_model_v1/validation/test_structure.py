
import pytest
from medins_ml_utils.utils import load_config
from pathlib import Path
import os

def test_config_exists():
    # Assumes running from the model directory or uses relative paths
    assert (Path(__file__).parent.parent / "config.json").exists()

def test_requirements_exists():
    assert (Path(__file__).parent.parent / "requirements.txt").exists()
