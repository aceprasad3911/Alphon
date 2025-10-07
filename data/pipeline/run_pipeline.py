"""
Master Pipeline Orchestrator
----------------------------
Coordinates all data pipeline phases for Alpha Signal Discovery.
"""

from src.ingestion import fetch_prices, fetch_fundamentals, fetch_esg
from src.staging import load_to_staging
from src.transformations import feature_engineering, graph_construction
from src.loading import load_features_to_db
from src.validation import validate_schema, validate_features
from src.monitoring import drift_monitor

import logging
import datetime

def main():
    logging.basicConfig(filename='pipeline.log', level=logging.INFO)
    start_time = datetime.datetime.now()
    logging.info(f"Pipeline started at {start_time}")

    try:
        # PHASE 1: Ingestion
        fetch_prices.run()
        fetch_fundamentals.run()
        fetch_esg.run()

        # PHASE 2: Staging
        load_to_staging.run()

        # PHASE 3: Transformation
        feature_engineering.run()
        graph_construction.run()

        # PHASE 4: Validation
        validate_schema.run()
        validate_features.run()

        # PHASE 5: Loading
        load_features_to_db.run()

        # PHASE 6: Monitoring
        drift_monitor.run()

        logging.info("✅ Pipeline completed successfully!")

    except Exception as e:
        logging.exception(f"❌ Pipeline failed: {e}")

    finally:
        end_time = datetime.datetime.now()
        logging.info(f"Pipeline finished at {end_time}")
        logging.info(f"Total runtime: {end_time - start_time}")

if __name__ == "__main__":
    main()
