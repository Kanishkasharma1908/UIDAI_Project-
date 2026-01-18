import requests
import pandas as pd

from config.api_config import (
    BASE_URL,
    API_KEY,
    API_KEY_DEMOGRAPHIC,
    API_KEY_BIOMETRIC,
    RAW_DATA_DIR,
    AADHAAR_ENROLMENT_RESOURCE_ID,
    AADHAAR_DEMOGRAPHIC_RESOURCE_ID,
    AADHAAR_BIOMETRIC_RESOURCE_ID,
    DEFAULT_API_PARAMS
)

from storage.csv_store import save_dataframe_to_csv
from utils.state_mapper import normalize_state_name


def fetch_state_dataset(
    resource_id: str,
    api_key: str,
    state: str,
    limit: int = 10000
) -> pd.DataFrame:
    """
    Fetch FULL Aadhaar dataset for a given state using pagination
    """
    all_records = []
    offset = 0

    while True:
        params = {
            **DEFAULT_API_PARAMS,
            "api-key": api_key,
            "filters[state]": state,
            "limit": limit,
            "offset": offset
        }

        url = f"{BASE_URL}/{resource_id}"
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        records = data.get("records", [])

        if not records:
            break  

        all_records.extend(records)
        offset += limit

        print(f"Fetched {len(all_records)} rows for {state} (resource={resource_id})")

    if not all_records:
        raise ValueError(f"No data found for state: {state}")

    return pd.DataFrame(all_records)


def fetch_and_store_single_state_data(state: str) -> dict:
    """
    Fetches enrolment, demographic, and biometric data for ONE state
    and stores them under Data/Raw/<dataset>/
    """

    normalized_state = normalize_state_name(state)

    datasets = {
        "enrolment": {
            "resource_id": AADHAAR_ENROLMENT_RESOURCE_ID,
            "api_key": API_KEY
        },
        "demographic": {
            "resource_id": AADHAAR_DEMOGRAPHIC_RESOURCE_ID,
            "api_key": API_KEY_DEMOGRAPHIC
        },
        "biometric": {
            "resource_id": AADHAAR_BIOMETRIC_RESOURCE_ID,
            "api_key": API_KEY_BIOMETRIC
        }
    }

    results = {}

    for dataset_name, cfg in datasets.items():
        print(f"\nFetching {dataset_name} data for {state}...")

        df = fetch_state_dataset(
            resource_id=cfg["resource_id"],
            api_key=cfg["api_key"],
            state=state
        )

        save_dataframe_to_csv(
            df=df,
            folder_path=RAW_DATA_DIR / dataset_name,
            file_name=f"{normalized_state}.csv"
        )

        results[dataset_name] = {
            "rows": len(df),
            "file": RAW_DATA_DIR / dataset_name / f"{normalized_state}.csv"
        }

        print(f"Saved {len(df)} rows → {dataset_name}/{normalized_state}.csv")

        #  free memory immediately
        del df

    return results

def fetch_and_store_states_data(states: list[str]) -> dict:
    """
    Fetches and stores Aadhaar datasets for multiple states sequentially
    """

    all_results = {}

    for state in states:
        print(f"\n==============================")
        print(f"PROCESSING STATE: {state.upper()}")
        print(f"==============================")

        state_result = fetch_and_store_single_state_data(state)
        all_results[state] = state_result

    return all_results
