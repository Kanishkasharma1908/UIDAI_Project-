

import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "Data" / "Raw"


# Base URL for data.gov.in API
BASE_URL = "https://api.data.gov.in/resource"

# Dataset resource IDs
AADHAAR_ENROLMENT_RESOURCE_ID = "ecd49b12-3084-4521-8f7e-ca8bf72069ba"
AADHAAR_DEMOGRAPHIC_RESOURCE_ID = "19eac040-0b94-49fa-b239-4f2fd8677d53"
AADHAAR_BIOMETRIC_RESOURCE_ID = "65454dab-1517-40a3-ac1d-47d4dfe6891c"

# API keys from environment
API_KEY = os.getenv("API_KEY")
API_KEY_DEMOGRAPHIC = os.getenv("API_KEY_DEMOGRAPHIC")
API_KEY_BIOMETRIC = os.getenv("API_KEY_BIOMETRIC")

# Fail fast if missing
if API_KEY is None:
    raise EnvironmentError("API_KEY (enrolment) not found in environment variables")

if API_KEY_DEMOGRAPHIC is None:
    raise EnvironmentError("API_KEY_DEMOGRAPHIC not found in environment variables")

if API_KEY_BIOMETRIC is None:
    raise EnvironmentError("API_KEY_BIOMETRIC not found in environment variables")

# Default API parameters
DEFAULT_API_PARAMS = {
    "format": "json",
    "offset": 0
}

PROCESSED_DATA_DIR = RAW_DATA_DIR.parent / "Processed"  # e.g., Data/Processed/
REPORTS_DIR = Path("Reports")  # e.g., Reports/
EDA_PLOTS_DIR = REPORTS_DIR / "EDA_ENROLMENT"  # e.g., Reports/EDA_Plots/
