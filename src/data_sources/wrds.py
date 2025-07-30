# src/data_sources/wrds.py

# Handles data fetching for WRDS API

import wrds
import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class WRDSSource(DataSource):
    """
    Data source handler for WRDS (Wharton Research Data Services).
    Requires institutional access and WRDS account credentials.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.username = self.config.get("username")
        self.password = self.config.get("password")
        self.default_library = self.config.get("default_library", "crsp")
        self.default_table = self.config.get("default_table", "msf") # Monthly Stock File

        self.db = None # WRDS connection object

    def _connect(self):
        """Establishes a connection to the WRDS database."""
        if self.db is None or not self.db.connected:
            logger.info("Attempting to connect to WRDS...")
            try:
                # WRDS library handles connection details, but can be overridden
                self.db = wrds.Connection(
                    wrds_username=self.username,
                    wrds_password=self.password,
                    # host=self.config.get("host"), # Uncomment if custom host/port needed
                    # port=self.config.get("port")
                )
                if self.db.connected:
                    logger.info("Successfully connected to WRDS.")
                else:
                    logger.error("Failed to establish WRDS connection.")
                    raise ConnectionError("WRDS connection failed.")
            except Exception as e:
                logger.error(f"Error connecting to WRDS: {e}")
                raise

    def _disconnect(self):
        """Closes the connection to the WRDS database."""
        if self.db and self.db.connected:
            logger.info("Closing WRDS connection.")
            self.db.close()
            self.db = None

    def fetch(self, library: str = None, table: str = None, query: str = None,
              columns: Optional[list] = None, **kwargs) -> pd.DataFrame:
        """
        Fetches data from WRDS using either a direct query or library/table specification.
        Args:
            library (str): WRDS library name (e.g., "crsp", "comp"). Defaults to config.
            table (str): Table name within the library (e.g., "msf", "dsf"). Defaults to config.
            query (str): Optional raw SQL query. If provided, library and table are ignored.
            columns (Optional[list]): List of columns to select.
            **kwargs: Additional parameters for the WRDS query (e.g., date filters).
        Returns:
            pd.DataFrame: Data fetched from WRDS.
        """
        self._connect() # Ensure connection is open

        library = library if library is not None else self.default_library
        table = table if table is not None else self.default_table

        logger.info(f"Fetching data from WRDS: Library='{library}', Table='{table}' (or custom query).")
        try:
            if query:
                data = self.db.raw_sql(query, **kwargs)
            else:
                # Example of how to build a query for a specific table
                # TODO: Make this more flexible for different table types and filters
                sql_query = f"SELECT "
                if columns:
                    sql_query += ", ".join(columns)
                else:
                    sql_query += "*" # Select all columns if not specified
                sql_query += f" FROM {library}.{table}"

                # Add date filters if provided in kwargs
                if 'start_date' in kwargs and 'end_date' in kwargs:
                    # Assuming a 'date' or 'caldt' column for filtering
                    date_col = kwargs.get('date_column', 'date') # Default date column name
                    sql_query += f" WHERE {date_col} BETWEEN '{kwargs['start_date']}' AND '{kwargs['end_date']}'"
                # TODO: Add more WHERE clauses for other filters (e.g., permno, gvkey)

                data = self.db.raw_sql(sql_query)

            if data.empty:
                logger.warning(f"No data found for WRDS query: {query or f'{library}.{table}'}.")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data from WRDS: {e}")
            raise
        finally:
            self._disconnect() # Close connection after fetching

    def normalize(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes WRDS data.
        This is highly dependent on the specific WRDS table queried.
        Args:
            raw_data (pd.DataFrame): Raw data from WRDS.
        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        if raw_data.empty:
            return pd.DataFrame()

        # Example normalization for CRSP daily stock file (dsf) or monthly (msf)
        # Common columns: permno, date, prc, ret, vol, shrout, bid, ask
        # TODO: Implement specific normalization logic based on the WRDS table being queried.
        # This might involve:
        # - Renaming columns (e.g., 'PRC' to 'price', 'RET' to 'return')
        # - Converting date columns to datetime objects
        # - Handling missing values specific to WRDS data
        # - Filtering out non-common share classes or delisted securities

        normalized_data = raw_data.copy()
        normalized_data.columns = [col.lower() for col in normalized_data.columns]

        if 'date' in normalized_data.columns:
            normalized_data['date'] = pd.to_datetime(normalized_data['date'])
            normalized_data = normalized_data.set_index('date').sort_index()
        elif 'caldt' in normalized_data.columns: # Common in some CRSP tables
            normalized_data['caldt'] = pd.to_datetime(normalized_data['caldt'])
            normalized_data = normalized_data.set_index('caldt').sort_index()
            normalized_data = normalized_data.rename_axis('date') # Rename index to 'date'

        logger.debug("WRDS data normalized (basic).")
        return normalized_data

# TODO: Implement more specific fetch methods for common WRDS datasets (e.g., CRSP, Compustat).
# TODO: Add robust error handling for WRDS connection issues and query errors.
# TODO: Consider using a context manager for WRDS connection to ensure it's always closed.
