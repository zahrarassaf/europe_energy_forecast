import pandas as pd

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard time-based features used in forecasting models.
    """
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday
    df['month'] = df['timestamp'].dt.month
    return df

def create_supervised(df: pd.DataFrame, target_col: str, n_lags: int = 24) -> pd.DataFrame:
    """
    Transform a time-series into a supervised-learning dataset.
    """
    data = df.copy()
    for i in range(1, n_lags + 1):
        data[f'{target_col}_lag_{i}'] = data[target_col].shift(i)

    data = data.dropna()
    return data
