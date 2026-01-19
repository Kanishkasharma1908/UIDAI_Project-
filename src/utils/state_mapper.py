def normalize_state_name(state_name: str) -> str:
    """
    Standardizes all 28 Indian States and 8 Union Territories.
    Maps noisy variations to official names for consistent file naming.
    """
    # 1. Clean basic white space and casing
    name_clean = state_name.strip().upper()
    
    # 2. Comprehensive mapping of variations found in your datasets
    mapping = {
        # States
        "ANDHRA PRADESH": "ANDHRA PRADESH",
        "ARUNACHAL PRADESH": "ARUNACHAL PRADESH",
        "ASSAM": "ASSAM",
        "BIHAR": "BIHAR",
        "CHHATTISGARH": "CHHATTISGARH",
        "CHATTISGARH": "CHHATTISGARH",
        "GOA": "GOA",
        "GUJARAT": "GUJARAT",
        "HARYANA": "HARYANA",
        "HIMACHAL PRADESH": "HIMACHAL PRADESH",
        "JHARKHAND": "JHARKHAND",
        "KARNATAKA": "KARNATAKA",
        "KERALA": "KERALA",
        "MADHYA PRADESH": "MADHYA PRADESH",
        "MAHARASHTRA": "MAHARASHTRA",
        "MANIPUR": "MANIPUR",
        "MEGHALAYA": "MEGHALAYA",
        "MIZORAM": "MIZORAM",
        "NAGALAND": "NAGALAND",
        "ODISHA": "ODISHA",
        "ORISSA": "ODISHA",
        "PUNJAB": "PUNJAB",
        "RAJASTHAN": "RAJASTHAN",
        "SIKKIM": "SIKKIM",
        "TAMIL NADU": "TAMIL NADU",
        "TAMILNADU": "TAMIL NADU",
        "TELANGANA": "TELANGANA",
        "TELENGANA": "TELANGANA",
        "TRIPURA": "TRIPURA",
        "UTTAR PRADESH": "UTTAR PRADESH",
        "UTTARAKHAND": "UTTARAKHAND",
        "UTTARANCHAL": "UTTARAKHAND",
        "UTTARANACHAL": "UTTARAKHAND",
        "WEST BENGAL": "WEST BENGAL",
        "WESTBENGAL": "WEST BENGAL",
        "WEST BANGAL": "WEST BENGAL",
        "WEST BENGALI": "WEST BENGAL",
        "WEST BENGLI": "WEST BENGAL",

        # Union Territories
        "ANDAMAN AND NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
        "ANDAMAN & NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
        "CHANDIGARH": "CHANDIGARH",
        "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
        "DADRA & NAGAR HAVELI": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
        "DAMAN & DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
        "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
        "DELHI": "DELHI",
        "NATIONAL CAPITAL TERRITORY OF DELHI": "DELHI",
        "JAMMU AND KASHMIR": "JAMMU AND KASHMIR",
        "JAMMU & KASHMIR": "JAMMU AND KASHMIR",
        "LADAKH": "LADAKH",
        "LAKSHADWEEP": "LAKSHADWEEP",
        "PUDUCHERRY": "PUDUCHERRY",
        "PONDICHERRY": "PUDUCHERRY",

        # Observed City/Regional Level Typos
        "BALANAGAR": "TELANGANA",
        "GURGAON": "HARYANA",
        "JAIPUR": "RAJASTHAN",
        "MADANAPALLE": "ANDHRA PRADESH",
        "NAGPUR": "MAHARASHTRA",
        "PUNE CITY": "MAHARASHTRA"
    }
    
    # 3. Apply mapping or default to cleaned original
    standard_name = mapping.get(name_clean, name_clean)
    
    # 4. Return lower_case_with_underscore for file safety
    return (
        standard_name
        .lower()
        .replace(" ", "_")
        .replace("&", "and")
    )