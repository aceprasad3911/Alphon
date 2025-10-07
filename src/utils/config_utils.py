# src/utils/config_loader.py

# Utility to load and parse configuration files (e.g., YAML)

import yaml
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        Dict[str, Any]: Loaded configuration as a dictionary.
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config {config_path}: {e}")
        raise

def load_db_config():
    cfg_path = ROOT / "config" / "db_config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    db = cfg.get("db", {})
    # Allow env var overrides
    db['user'] = os.getenv("DB_USER", db.get('user'))
    db['password'] = os.getenv("DB_PASSWORD", db.get('password'))
    db['host'] = os.getenv("DB_HOST", db.get('host'))
    db['port'] = int(os.getenv("DB_PORT", db.get('port')))
    db['dbname'] = os.getenv("DB_NAME", db.get('dbname', db.get('dbname')))
    return db

def load_data_sources():
    cfg_path = ROOT / "config" / "data_sources.json"
    with open(cfg_path, "r") as f:
        return json.load(f)

def load_all_configs(config_dir: str = "config/") -> Dict[str, Any]:
    """
    Loads all YAML configuration files from a specified directory.
    Args:
        config_dir (str): Path to the directory containing YAML config files.
    Returns:
        Dict[str, Any]: A dictionary where keys are file names (without extension)
                        and values are the loaded configurations.
    """
    all_configs = {}
    if not os.path.isdir(config_dir):
        logger.warning(f"Config directory not found: {config_dir}. Returning empty configs.")
        return all_configs

    for filename in os.listdir(config_dir):
        if filename.endswith((".yaml", ".yml")):
            file_path = os.path.join(config_dir, filename)
            config_name = os.path.splitext(filename)[0]
            try:
                all_configs[config_name] = load_config(file_path)
            except Exception as e:
                logger.error(f"Failed to load config file {filename}: {e}")
                # Continue to load other files even if one fails
    logger.info(f"Loaded {len(all_configs)} configuration files from {config_dir}.")
    return all_configs

# TODO: Implement a function to validate loaded configurations against a schema.
# TODO: Add support for environment variables for sensitive data (e.g., API keys).
