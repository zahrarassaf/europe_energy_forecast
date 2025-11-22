import os
import pandas as pd

def load_raw_csv(path: str) -> pd.DataFrame:
    """
    Load a raw CSV file and return a pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def save_processed(df: pd.DataFrame, path: str) -> None:
    """
    Save a processed dataframe into the processed folder.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_processed(path: str) -> pd.DataFrame:
    """
    Load a processed (clean) dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path)
