def normalize_state_name(state_name: str) -> str:
    """
    Converts state name to file-safe format
    Example: 'Tamil Nadu' -> 'tamil_nadu'
    """
    return (
        state_name
        .strip()
        .lower()
        .replace(" ", "_")
    )
