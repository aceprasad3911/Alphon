import os
import subprocess
import sys

def run_phase_1_pipeline():
    """Orchestrates the execution of Phase 1: Foundation & Data Engineering."""
    print("--- Starting Phase 1: Foundation & Data Engineering ---")

    # 1. Run Data Acquisition
    print("\n--- Running Data Acquisition (src/data_acquisition.py) ---")
    try:
        # Use sys.executable to ensure the script runs with the active virtual environment's python
        subprocess.run([sys.executable, 'src/data_acquisition.py'], check=True)
        print("Data Acquisition completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Data Acquisition: {e}")
        print("Please ensure your FRED_API_KEY is set correctly in data_acquisition.py and you have internet access.")
        return

    # 2. Run Data Cleaning and Alignment
    print("\n--- Running Data Cleaning & Alignment (src/data_cleaning.py) ---")
    try:
        subprocess.run([sys.executable, 'src/data_cleaning.py'], check=True)
        print("Data Cleaning & Alignment completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Data Cleaning & Alignment: {e}")
        print("Please check raw data files and cleaning logic.")
        return

    # 3. Run Feature Engineering
    print("\n--- Running Feature Engineering (src/feature_engineering.py) ---")
    try:
        subprocess.run([sys.executable, 'src/feature_engineering.py'], check=True)
        print("Feature Engineering completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Feature Engineering: {e}")
        print("Please check processed data files and feature engineering logic.")
        return

    print("\n--- Phase 1: Foundation & Data Engineering Complete! ---")
    print("You should now have processed data and initial features in the 'data/processed' and 'data/features' directories.")

if __name__ == "__main__":
    # Ensure you are in the project root directory when running this script
    # e.g., from my_quant_project/ run: python src/main_pipeline.py
    run_phase_1_pipeline()
