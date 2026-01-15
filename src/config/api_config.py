import os

# Base URL for data.gov.in API
BASE_URL = "https://api.data.gov.in/resource"


# Aadhaar enrolment dataset resource ID
AADHAAR_ENROLMENT_RESOURCE_ID = "ecd49b12-3084-4521-8f7e-ca8bf72069ba"

# Read API key from environment variable
API_KEY = os.getenv("API_KEY")

if API_KEY is None:
    raise EnvironmentError(
        "API_KEY not found in environment variables"
    )

# Default API parameters
DEFAULT_API_PARAMS = {
    "format": "json",
    "limit": 100000
}
