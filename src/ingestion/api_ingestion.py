import time
import requests
import pandas as pd
import logging
import gc
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIG IMPORTS ---
from config.api_config import (
    BASE_URL, API_KEY, API_KEY_DEMOGRAPHIC, API_KEY_BIOMETRIC,
    RAW_DATA_DIR, AADHAAR_ENROLMENT_RESOURCE_ID,
    AADHAAR_DEMOGRAPHIC_RESOURCE_ID, AADHAAR_BIOMETRIC_RESOURCE_ID,
    DEFAULT_API_PARAMS
)
from utils.state_mapper import normalize_state_name

def get_resilient_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def save_chunk_by_state(chunk_df: pd.DataFrame, dataset_type: str):
    """Splits chunk and appends to existing CSVs without losing previous data."""
    for state_name, group in chunk_df.groupby('state'):
        file_safe = normalize_state_name(str(state_name))
        folder_path = RAW_DATA_DIR / dataset_type
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / f"{file_safe}.csv"
        
        # Append mode ('a') ensures we don't delete what you fetched yesterday
        is_new_file = not file_path.exists()
        group.to_csv(file_path, mode='a', index=False, header=is_new_file)

def harvest_and_split_resource(resource_id: str, api_key: str, name: str, start_offset: int = 0):
    """Fetches records globally and saves state-wise CSVs in chunks."""
    offset = start_offset
    chunk_size = 100 
    session = get_resilient_session()
    buffer_records = []

    while True:
        params = {
            "api-key": api_key,
            "format": "json",
            "limit": chunk_size,
            "offset": offset
        }
        
        try:
            url = f"{BASE_URL}/{resource_id}"
            response = session.get(url, params=params, timeout=30)
            data = response.json()
            records = data.get("records", [])
            
            if not records:
                if buffer_records:
                    save_chunk_by_state(optimize_dataframe(pd.DataFrame(buffer_records)), name)
                break
                
            buffer_records.extend(records)
            offset += chunk_size
            
            # Save every 5,000 records to disk to keep RAM healthy
            if len(buffer_records) >= 5000:
                save_chunk_by_state(optimize_dataframe(pd.DataFrame(buffer_records)), name)
                buffer_records = []
                gc.collect()

            total = data.get('total', 'unknown')
            if offset % 1000 == 0:
                logger.info(f"📥 [{name.upper()}] Progress: {offset} / {total} records...")

        except Exception as e:
            logger.error(f" Error in {name} at offset {offset}: {e}")
            break

def run_full_pipeline(resume_enrolment_at=0):
    """
    Triggers the harvest for all 3 datasets.
    Pass resume_enrolment_at=370300 to pick up where you left off.
    """
    datasets = [
        {"name": "enrolment", "id": AADHAAR_ENROLMENT_RESOURCE_ID, "key": API_KEY},
        {"name": "demographic", "id": AADHAAR_DEMOGRAPHIC_RESOURCE_ID, "key": API_KEY_DEMOGRAPHIC},
        {"name": "biometric", "id": AADHAAR_BIOMETRIC_RESOURCE_ID, "key": API_KEY_BIOMETRIC}
    ]

    for ds in datasets:
        logger.info(f"\n--- Starting {ds['name'].upper()} ---")
        
        # Use the resume offset only for the interrupted 'enrolment' set
        offset_to_use = resume_enrolment_at if ds['name'] == "enrolment" else 0
        
        if offset_to_use > 0:
            logger.info(f" Resuming Enrolment from {offset_to_use}...")

        harvest_and_split_resource(
            resource_id=ds['id'], 
            api_key=ds['key'], 
            name=ds['name'], 
            start_offset=offset_to_use
        )