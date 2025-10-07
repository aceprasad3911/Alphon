# fred.py

# https://fredaccount.stlouisfed.org/apikeys

# Formatted Data Calls from API


import quandl
import yaml
from pathlib import Path

# Return to base directory (Alphon) by finding parent directories (x4)
base_dir = Path(__file__).resolve().parent.parent.parent.parent
# Define the path to the config directory and the YAML file
config_path = base_dir / 'config' / 'api_keys.yaml'
# Load the API key from the YAML file
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
# Access the API key
fred_api_key = config['fred']['api_key']
# Now use the API key in API calls
print(f"FRED API Key:{fred_api_key}")