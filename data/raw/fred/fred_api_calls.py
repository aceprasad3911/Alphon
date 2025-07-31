# fred_api_calls.py

# Raw Data Calls from API

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

####################################
### 1. Category API Calls ##########
####################################
import requests

# Get Category ---------------------------------------------------------------------------------
def get_category(category_id):
    url = f'https://api.stlouisfed.org/fred/category?category_id={category_id}&api_key={fred_api_key}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
get_category(125)  # Replace with desired category_id

# Get Child Categories -------------------------------------------------------------------------
def get_child_categories(parent_category_id):
    url = f'https://api.stlouisfed.org/fred/category/children?category_id={parent_category_id}&api_key={fred_api_key}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
get_child_categories(13)  # Replace with desired parent_category_id

# Get Related Categories ----------------------------------------------------------------------
def get_related_categories(category_id):
    url = f'https://api.stlouisfed.org/fred/category/related?category_id={category_id}&api_key={fred_api_key}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
get_related_categories(32073)  # Replace with desired category_id

# Get Series in a Category -------------------------------------------------------------------
def get_series_in_category(category_id):
    url = f'https://api.stlouisfed.org/fred/category/series?category_id={category_id}&api_key={fred_api_key}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
get_series_in_category(125)  # Replace with desired category_id

# Get Tags for a Category ---------------------------------------------------------------------
def get_tags_for_category(category_id):
    url = f'https://api.stlouisfed.org/fred/category/tags?category_id={category_id}&api_key={fred_api_key}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
get_tags_for_category(125)  # Replace with desired category_id

# Get Related Tags for a Category -------------------------------------------------------------
def get_related_tags(category_id, tag_names):
    url = f'https://api.stlouisfed.org/fred/category/related_tags?category_id={category_id}&tag_names={tag_names}&api_key={fred_api_key}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
get_related_tags(125, 'services;quarterly')  # Replace with desired tag_names

#############################
### 2. Releases API Calls ###
#############################
import requests

# Get Releases -------------------------------------------------------------------------------
def fetch_releases():
    url = f'https://api.stlouisfed.org/fred/releases?api_key={fred_api_key}&file_type=json'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_releases()  # Call to get all releases

# Get Release Metadata ----------------------------------------------------------------------
def fetch_release_metadata(release_id):
    url = f'https://api.stlouisfed.org/fred/release?release_id={release_id}&api_key={fred_api_key}&file_type=json'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_release_metadata(53)  # Replace with desired release_id

# Get Release Dates -------------------------------------------------------------------------
def fetch_release_dates(release_id):
    url = f'https://api.stlouisfed.org/fred/release/dates?release_id={release_id}&api_key={fred_api_key}&file_type=json'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_release_dates(82)  # Replace with desired release_id

# Get Series in a Release ------------------------------------------------------------------
def fetch_series_in_release(release_id):
    url = f'https://api.stlouisfed.org/fred/release/series?release_id={release_id}&api_key={fred_api_key}&file_type=json'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_series_in_release(51)  # Replace with desired release_id

# Get Sources for a Release ---------------------------------------------------------------
def fetch_sources_for_release(release_id):
    url = f'https://api.stlouisfed.org/fred/release/sources?release_id={release_id}&api_key={fred_api_key}&file_type=json'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_sources_for_release(51)  # Replace with desired release_id

# Get Tags for a Release --------------------------------------------------------------------
def fetch_tags_for_release(release_id):
    url = f'https://api.stlouisfed.org/fred/release/tags?release_id={release_id}&api_key={fred_api_key}&file_type=json'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_tags_for_release(86)  # Replace with desired release_id

# Get Related Tags for a Release -----------------------------------------------------------
def fetch_related_tags_for_release(release_id, tag_names):
    url = f'https://api.stlouisfed.org/fred/release/related_tags?release_id={release_id}&tag_names={tag_names}&api_key={fred_api_key}&file_type=json'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_related_tags_for_release(86, 'sa;foreign')  # Replace with desired release_id and tag_names

# Get Release Tables ------------------------------------------------------------------------
def fetch_release_tables(release_id, element_id=None):
    url = f'https://api.stlouisfed.org/fred/release/tables?release_id={release_id}&api_key={fred_api_key}&file_type=json'
    if element_id:
        url += f'&element_id={element_id}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_release_tables(53, 12886)  # Replace with desired release_id and element_id

#############################
### 3. Series API Calls ###
#############################
import requests

# Get an Economic Data Series ----------------------------------------------------------------
def fetch_series(series_id, realtime_start=None, realtime_end=None):
    url = f'https://api.stlouisfed.org/fred/series?series_id={series_id}&api_key={fred_api_key}&file_type=json'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_series('GNPCA')
# fetch_series('GNPCA', realtime_start='2013-01-01', realtime_end='2013-12-31')

# Get Categories for a Series ---------------------------------------------------------------
def fetch_series_categories(series_id, realtime_start=None, realtime_end=None):
    url = f'https://api.stlouisfed.org/fred/series/categories?series_id={series_id}&api_key={fred_api_key}&file_type=json'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_series_categories('EXJPUS')

# Get Observations for a Series -------------------------------------------------------------
def fetch_series_observations(series_id, realtime_start=None, realtime_end=None, limit=None, offset=None, sort_order=None, observation_start=None, observation_end=None, units=None, frequency=None, aggregation_method=None, output_type=None, vintage_dates=None):
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={fred_api_key}&file_type=json'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    if limit:
        url += f'&limit={limit}'
    if offset:
        url += f'&offset={offset}'
    if sort_order:
        url += f'&sort_order={sort_order}'
    if observation_start:
        url += f'&observation_start={observation_start}'
    if observation_end:
        url += f'&observation_end={observation_end}'
    if units:
        url += f'&units={units}'
    if frequency:
        url += f'&frequency={frequency}'
    if aggregation_method:
        url += f'&aggregation_method={aggregation_method}'
    if output_type:
        url += f'&output_type={output_type}'
    if vintage_dates:
        url += f'&vintage_dates={vintage_dates}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_series_observations('GNPCA', limit=5)
# fetch_series_observations('GNPCA', observation_start='2000-01-01', observation_end='2010-12-31', units='pch', frequency='a')

# Get Release for a Series ------------------------------------------------------------------
def fetch_series_release(series_id, realtime_start=None, realtime_end=None):
    url = f'https://api.stlouisfed.org/fred/series/release?series_id={series_id}&api_key={fred_api_key}&file_type=json'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_series_release('IRA')

# Search for Series -------------------------------------------------------------------------
def search_series(search_text, search_type=None, realtime_start=None, realtime_end=None, limit=None, offset=None, order_by=None, sort_order=None, filter_variable=None, filter_value=None, tag_names=None, exclude_tag_names=None):
    url = f'https://api.stlouisfed.org/fred/series/search?search_text={search_text}&api_key={fred_api_key}&file_type=json'
    if search_type:
        url += f'&search_type={search_type}'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    if limit:
        url += f'&limit={limit}'
    if offset:
        url += f'&offset={offset}'
    if order_by:
        url += f'&order_by={order_by}'
    if sort_order:
        url += f'&sort_order={sort_order}'
    if filter_variable:
        url += f'&filter_variable={filter_variable}'
    if filter_value:
        url += f'&filter_value={filter_value}'
    if tag_names:
        url += f'&tag_names={tag_names}'
    if exclude_tag_names:
        url += f'&exclude_tag_names={exclude_tag_names}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
search_series('monetary service index', limit=3)
# search_series('GNP*', search_type='series_id', sort_order='asc')

# Get Tags for a Series Search --------------------------------------------------------------
def fetch_series_search_tags(series_search_text, realtime_start=None, realtime_end=None, tag_names=None, tag_group_id=None, tag_search_text=None, limit=None, offset=None, order_by=None, sort_order=None):
    url = f'https://api.stlouisfed.org/fred/series/search/tags?series_search_text={series_search_text}&api_key={fred_api_key}&file_type=json'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    if tag_names:
        url += f'&tag_names={tag_names}'
    if tag_group_id:
        url += f'&tag_group_id={tag_group_id}'
    if tag_search_text:
        url += f'&tag_search_text={tag_search_text}'
    if limit:
        url += f'&limit={limit}'
    if offset:
        url += f'&offset={offset}'
    if order_by:
        url += f'&order_by={order_by}'
    if sort_order:
        url += f'&sort_order={sort_order}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_series_search_tags('monetary service index', limit=3)
# fetch_series_search_tags('mortgage rate', tag_group_id='freq')

# Get Related Tags for a Series Search ------------------------------------------------------
def fetch_series_search_related_tags(series_search_text, tag_names, realtime_start=None, realtime_end=None, exclude_tag_names=None, tag_group_id=None, tag_search_text=None, limit=None, offset=None, order_by=None, sort_order=None):
    url = f'https://api.stlouisfed.org/fred/series/search/related_tags?series_search_text={series_search_text}&tag_names={tag_names}&api_key={fred_api_key}&file_type=json'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    if exclude_tag_names:
        url += f'&exclude_tag_names={exclude_tag_names}'
    if tag_group_id:
        url += f'&tag_group_id={tag_group_id}'
    if tag_search_text:
        url += f'&tag_search_text={tag_search_text}'
    if limit:
        url += f'&limit={limit}'
    if offset:
        url += f'&offset={offset}'
    if order_by:
        url += f'&order_by={order_by}'
    if sort_order:
        url += f'&sort_order={sort_order}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_series_search_related_tags('mortgage rate', '30-year;frb')
# fetch_series_search_related_tags('mortgage rate', '30-year;frb', exclude_tag_names='discontinued;monthly')

# Get Tags for a Series ---------------------------------------------------------------------
def fetch_series_tags(series_id, realtime_start=None, realtime_end=None, order_by=None, sort_order=None):
    url = f'https://api.stlouisfed.org/fred/series/tags?series_id={series_id}&api_key={fred_api_key}&file_type=json'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    if order_by:
        url += f'&order_by={order_by}'
    if sort_order:
        url += f'&sort_order={sort_order}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_series_tags('STLFSI')

# Get Series Updates ------------------------------------------------------------------------
def fetch_series_updates(realtime_start=None, realtime_end=None, limit=None, offset=None, filter_value=None, start_time=None, end_time=None):
    url = f'https://api.stlouisfed.org/fred/series/updates?api_key={fred_api_key}&file_type=json'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    if limit:
        url += f'&limit={limit}'
    if offset:
        url += f'&offset={offset}'
    if filter_value:
        url += f'&filter_value={filter_value}'
    if start_time:
        url += f'&start_time={start_time}'
    if end_time:
        url += f'&end_time={end_time}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_series_updates(limit=3)
# fetch_series_updates(filter_value='regional')

# Get Vintage Dates for a Series ------------------------------------------------------------
def fetch_series_vintagedates(series_id, realtime_start=None, realtime_end=None, limit=None, offset=None, sort_order=None):
    url = f'https://api.stlouisfed.org/fred/series/vintagedates?series_id={series_id}&api_key={fred_api_key}&file_type=json'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    if limit:
        url += f'&limit={limit}'
    if offset:
        url += f'&offset={offset}'
    if sort_order:
        url += f'&sort_order={sort_order}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_series_vintagedates('GNPCA', limit=5)

#############################
### 4. Tags API Calls ###
#############################
import requests

# Get FRED Tags -------------------------------------------------------------------------------
def fetch_tags(realtime_start=None, realtime_end=None, tag_names=None, tag_group_id=None, search_text=None, limit=None, offset=None, order_by=None, sort_order=None):
    url = f'https://api.stlouisfed.org/fred/tags?api_key={fred_api_key}&file_type=json'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    if tag_names:
        url += f'&tag_names={tag_names}'
    if tag_group_id:
        url += f'&tag_group_id={tag_group_id}'
    if search_text:
        url += f'&search_text={search_text}'
    if limit:
        url += f'&limit={limit}'
    if offset:
        url += f'&offset={offset}'
    if order_by:
        url += f'&order_by={order_by}'
    if sort_order:
        url += f'&sort_order={sort_order}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_tags(limit=5)

# Get Related Tags ---------------------------------------------------------------------------
def fetch_related_tags(tag_names, exclude_tag_names=None, tag_group_id=None, search_text=None, limit=None, offset=None, order_by=None, sort_order=None):
    url = f'https://api.stlouisfed.org/fred/related_tags?tag_names={tag_names}&api_key={fred_api_key}&file_type=json'
    if exclude_tag_names:
        url += f'&exclude_tag_names={exclude_tag_names}'
    if tag_group_id:
        url += f'&tag_group_id={tag_group_id}'
    if search_text:
        url += f'&search_text={search_text}'
    if limit:
        url += f'&limit={limit}'
    if offset:
        url += f'&offset={offset}'
    if order_by:
        url += f'&order_by={order_by}'
    if sort_order:
        url += f'&sort_order={sort_order}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_related_tags('monetary aggregates', exclude_tag_names='discontinued', limit=5)

# Get Series by Tags ------------------------------------------------------------------------
def fetch_series_by_tags(tag_names, exclude_tag_names=None, realtime_start=None, realtime_end=None, limit=None, offset=None, order_by=None, sort_order=None):
    url = f'https://api.stlouisfed.org/fred/tags/series?tag_names={tag_names}&api_key={fred_api_key}&file_type=json'
    if exclude_tag_names:
        url += f'&exclude_tag_names={exclude_tag_names}'
    if realtime_start:
        url += f'&realtime_start={realtime_start}'
    if realtime_end:
        url += f'&realtime_end={realtime_end}'
    if limit:
        url += f'&limit={limit}'
    if offset:
        url += f'&offset={offset}'
    if order_by:
        url += f'&order_by={order_by}'
    if sort_order:
        url += f'&sort_order={sort_order}'
    r = requests.get(url)
    data = r.json()
    print(data)

# Example Usage
fetch_series_by_tags('slovenia;food', exclude_tag_names='alcohol', limit=5)
