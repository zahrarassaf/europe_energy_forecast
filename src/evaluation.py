# src/evaluation.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    # avoid division by zero
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def evaluate_all(y_true, y_pred):
    return {
        "RMSE": float(rmse(y_true, y_pred)),
        "MAE": float(mae(y_true, y_pred)),
        "MAPE": float(mape(y_true, y_pred))
    }
