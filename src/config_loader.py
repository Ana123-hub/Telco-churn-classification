import os
import yaml

# Find the project root dynamically relative to this script
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

def load_config():
    """Loads the YAML configuration file and injects resolved absolute paths."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Convert relative paths in config to safe absolute paths anchored to PROJECT_ROOT
    config["abs_paths"] = {
        "raw_data": os.path.join(PROJECT_ROOT, config["paths"]["raw_data"]),
        "processed_dir": os.path.join(PROJECT_ROOT, config["paths"]["processed_dir"]),
        "models_dir": os.path.join(PROJECT_ROOT, config["paths"]["models_dir"]),
        "project_root": PROJECT_ROOT
    }
    
    return config