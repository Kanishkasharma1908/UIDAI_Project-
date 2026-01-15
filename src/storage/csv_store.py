import os
import pandas as pd


print("csv_store loaded")



def save_dataframe_to_csv(
    df: pd.DataFrame,
    folder_path: str,
    file_name: str
) -> None:
    """
    Saves DataFrame to CSV at specified location
    """
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, file_name)
    df.to_csv(file_path, index=False)


def load_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads CSV into a pandas DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)
