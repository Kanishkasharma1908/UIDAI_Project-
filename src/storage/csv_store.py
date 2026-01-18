from pathlib import Path
import pandas as pd

print("csv_store loaded")


def save_dataframe_to_csv(
    df: pd.DataFrame,
    folder_path: Path,
    file_name: str
) -> None:
    """
    Saves DataFrame to CSV at specified location (absolute path only)
    """
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path = folder_path / file_name
    df.to_csv(file_path, index=False)


def load_csv_to_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Loads CSV into a pandas DataFrame
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)
