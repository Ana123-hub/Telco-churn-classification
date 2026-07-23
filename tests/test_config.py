import os
import pytest
from src.config_loader import load_config


def test_load_config_structure():
    """Verify that config loads as a dictionary with all required root sections."""
    config = load_config()
    
    assert isinstance(config, dict)
    assert "paths" in config
    assert "data_cleaning" in config
    assert "features" in config
    assert "model_params" in config
    assert "abs_paths" in config


def test_resolved_paths_exist_or_valid():
    """Ensure generated absolute paths are anchored to the project root."""
    config = load_config()
    abs_paths = config["abs_paths"]

    assert os.path.isabs(abs_paths["raw_data"])
    assert os.path.isabs(abs_paths["processed_dir"])
    assert os.path.isabs(abs_paths["models_dir"])
    assert os.path.exists(abs_paths["project_root"])


def test_feature_lists_not_empty():
    """Check that feature lists are populated in config."""
    config = load_config()
    
    assert len(config["features"]["numerical_cols"]) > 0
    assert len(config["features"]["categorical_cols"]) > 0