import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_baseline_predictions(df, target_col='DE_load_actual_entsoe_transparency'):
    """Create simple baseline predictions"""
    
    # Method 1: Previous day average (simplest baseline)
    df_sorted = df.sort_index()
    baseline_preds = df_sorted[target_col].shift(24)  # 24 hours ago
    
    # Method 2: Previous week average
    # baseline_preds = df_sorted[target_col].shift(24*7)
    
    return baseline_preds

def calculate_baseline_accuracy(df, target_col='DE_load_actual_entsoe_transparency'):
    """Calculate baseline accuracy"""
    
    # Create baseline predictions
    baseline_predictions = create_baseline_predictions(df, target_col)
    
    # Remove NaN values
    valid_mask = baseline_predictions.notna() & df[target_col].notna()
    y_true = df[target_col][valid_mask]
    y_pred = baseline_predictions[valid_mask]
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"Baseline Model Performance:")
    print(f"   MAE: {mae:.2f} MW")
    print(f"   RMSE: {rmse:.2f} MW") 
    print(f"   MAPE: {mape:.2f}%")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape}
