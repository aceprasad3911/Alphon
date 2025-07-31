# src/data_sourcing/edgar.py

# Handles data fetching for EDGAR API

import requests
import pandas as pd
import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class EDGARSource(DataSource):
    """
    Data source handler for SEC EDGAR API.
    Fetches company filings (e.g., 10-K, 10-Q).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = self.config.get("base_url", "https://data.sec.gov/")
        self.user_agent_email = self.config.get("user_agent_email")
        self.filing_types = self.config.get("filing_types", ["10-K", "10-Q"])
        self.default_cik = self.config.get("default_cik")

        if not self.user_agent_email:
            logger.warning("SEC EDGAR API requires a User-Agent header (your email). "
                           "Requests might be blocked without it.")
        self.headers = {'User-Agent': f'YourAppName {self.user_agent_email}'}

        # SEC recommends no more than 10 requests per second
        self.last_request_time = 0
        self.request_count_second = 0
        self.second_reset_time = time.time()

    def _apply_rate_limit(self):
        """Applies rate limiting for SEC EDGAR API."""
        current_time = time.time()

        if current_time - self.second_reset_time >= 1:
            self.request_count_second = 0
            self.second_reset_time = current_time

        if self.request_count_second >= 10:
            sleep_time = 1 - (current_time - self.second_reset_time) + 0.1 # Wait until next second
            logger.warning(f"EDGAR API second limit reached. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
            self.request_count_second = 0
            self.second_reset_time = time.time()

        self.request_count_second += 1
        self.last_request_time = current_time

    def fetch(self, cik: str = None, filing_type: str = None, limit: int = 10,
              start_date: str = None, end_date: str = None, **kwargs) -> pd.DataFrame:
        """
        Fetches company filings metadata from SEC EDGAR.
        Args:
            cik (str): Central Index Key (CIK) of the company (e.g., "0000320193" for Apple). Defaults to config.
            filing_type (str): Type of filing (e.g., "10-K", "10-Q"). Defaults to config.
            limit (int): Maximum number of filings to retrieve.
            start_date (str): Filter filings after this date (YYYY-MM-DD).
            end_date (str): Filter filings before this date (YYYY-MM-DD).
            **kwargs: Additional parameters (e.g., for specific API endpoints).
        Returns:
            pd.DataFrame: Metadata of fetched filings.
        """
        cik = cik if cik is not None else self.default_cik
        filing_type = filing_type if filing_type is not None else self.filing_types[0]

        if not cik:
            raise ValueError("CIK must be provided to fetch EDGAR filings.")

        # SEC CIKs are 10 digits, often padded with leading zeros
        cik_padded = str(cik).zfill(10)
        company_filings_url = f"{self.base_url}submissions/CIK{cik_padded}.json"

        self._apply_rate_limit()
        logger.info(f"Fetching filings for CIK {cik_padded} from EDGAR.")
        try:
            response = requests.get(company_filings_url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            # Parse recent filings
            recent_filings = data.get('filings', {}).get('recent', {})
            if not recent_filings:
                logger.warning(f"No recent filings found for CIK {cik_padded}.")
                return pd.DataFrame()

            filings_df = pd.DataFrame(recent_filings)

            # Filter by filing type
            if filing_type:
                filings_df = filings_df[filings_df['form'] == filing_type]

            # Filter by date
            filings_df['filingDate'] = pd.to_datetime(filings_df['filingDate'])
            if start_date:
                filings_df = filings_df[filings_df['filingDate'] >= pd.to_datetime(start_date)]
            if end_date:
                filings_df = filings_df[filings_df['filingDate'] <= pd.to_datetime(end_date)]

            # Limit number of results
            filings_df = filings_df.head(limit)

            return filings_df
        except requests.exceptions.RequestException as e:
            logger.error(f"Network or HTTP error fetching from EDGAR for CIK {cik_padded}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch EDGAR filings for CIK {cik_padded}: {e}")
            raise

    def download_filing_document(self, accession_number: str, primary_document: str) -> Optional[str]:
        """
        Downloads a specific filing document (e.g., 10-K text or XBRL).
        Args:
            accession_number (str): Accession number of the filing (e.g., "0000320193-23-000077").
            primary_document (str): Path to the primary document within the filing (e.g., "aapl-20230930.htm").
        Returns:
            Optional[str]: Content of the document as a string, or None if failed.
        """
        # Example URL: https://www.sec.gov/Archives/edgar/data/320193/0000320193-23-000077/aapl-20230930.htm
        # CIK is embedded in accession_number (first 10 digits)
        cik = accession_number.split('-')[0]
        accession_no_clean = accession_number.replace('-', '') # Remove hyphens for URL path

        document_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_clean}/{primary_document}"

        self._apply_rate_limit()
        logger.info(f"Downloading document: {document_url}")
        try:
            response = requests.get(document_url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading document {document_url}: {e}")
            return None

    def normalize(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes EDGAR filing metadata.
        Args:
            raw_data (pd.DataFrame): Raw filing metadata.
        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        if raw_data.empty:
            return pd.DataFrame()

        normalized_data = raw_data.copy()
        normalized_data.columns = [col.lower() for col in normalized_data.columns]

        # Ensure date columns are datetime objects
        if 'filingdate' in normalized_data.columns:
            normalized_data['filingdate'] = pd.to_datetime(normalized_data['filingdate'])
        if 'reportdate' in normalized_data.columns:
            normalized_data['reportdate'] = pd.to_datetime(normalized_data['reportdate'])

        # Set accessionNumber as index for easier lookup
        if 'accessionnumber' in normalized_data.columns:
            normalized_data = normalized_data.set_index('accessionnumber')

        logger.debug("EDGAR data normalized.")
        return normalized_data

# TODO: Implement parsing of XBRL data from filings for structured financial statements.
# TODO: Add a CIK lookup function (e.g., from ticker symbol).
# TODO: Handle different types of filings (e.g., 8-K, DEF 14A) and their specific parsing needs.
# TODO: Consider using a dedicated SEC API client library if available and more robust.
