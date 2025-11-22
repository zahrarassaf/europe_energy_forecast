import pandas as pd

def clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw OPSD dataset:
    - Parse datetime column
    - Handle missing values
    - Sort chronologically
    """
    df = df.copy()

    # Standard datetime parsing
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Forward-fill missing values (industry standard for energy data)
    df = df.ffill()

    return df

def filter_country(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """
    Filter dataset for a specific country.
    """
    return df[df['country'] == country_code].copy()
