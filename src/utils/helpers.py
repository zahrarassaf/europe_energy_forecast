import pandas as pd
import numpy as np

def debug_info(df, message=""):
    """Print debug information"""
    print(f"🔍 DEBUG {message}:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Index: {df.index[:3]}...")
    print()

def safe_feature_creation(df, config):
    """Safe feature creation with error handling"""
    try:
        # Basic features
        df = df.copy()
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        
        # Safe week calculation
        try:
            df['week_of_year'] = df.index.isocalendar().week.astype(int)
        except:
            df['week_of_year'] = (df.index.dayofyear / 7).astype(int)
        
        return df
    except Exception as e:
        print(f"Error in feature creation: {e}")
        return df
