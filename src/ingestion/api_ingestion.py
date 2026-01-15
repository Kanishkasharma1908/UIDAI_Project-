import requests
import pandas as pd

from config.api_config import (
    BASE_URL,
    API_KEY,
    AADHAAR_ENROLMENT_RESOURCE_ID,
    DEFAULT_API_PARAMS
)
from storage.csv_store import save_dataframe_to_csv
from utils.state_mapper import normalize_state_name


def fetch_state_enrolment_data(state_name: str) -> pd.DataFrame:
    """
    Fetch Aadhaar enrolment data for a specific state from data.gov.in API
    """
    params = {
        **DEFAULT_API_PARAMS,
        "api-key": API_KEY,
        "filters[state_name]": state_name
    }

    url = f"{BASE_URL}/{AADHAAR_ENROLMENT_RESOURCE_ID}"

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    if "records" not in data or len(data["records"]) == 0:
        raise ValueError(f"No data found for state: {state_name}")

    return pd.DataFrame(data["records"])


def fetch_and_store_state_data(state_name: str) -> pd.DataFrame:
    """
    Fetches state-wise data and stores it as CSV in data/raw/enrolment
    """
    df = fetch_state_enrolment_data(state_name)

    normalized_state = normalize_state_name(state_name)

    save_dataframe_to_csv(
        df=df,
        folder_path="data/raw/enrolment",
        file_name=f"{normalized_state}.csv"
    )

    return df
