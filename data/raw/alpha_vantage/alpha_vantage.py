# alpha_vantage.py

# Formatted Data Calls from API

# https://outlook.office.com/mail/id/AAQkADFjNzU5MzZmLTFhMDEtNDk5Mi05ODhjLTQ1NTgzZjgyMzZhNAAQANerZ2cYFd9OrOmi7wAnRzE%3D for BackUp API Key


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
alpha_vantage_api_key = config['alpha_vantage']['api_key']
# Now use the API key in API calls
print(f"Alpha Vantage API Key:{alpha_vantage_api_key}")