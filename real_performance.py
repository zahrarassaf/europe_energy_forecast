import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

def load_your_data():
    """Load your actual dataset"""
    if os.path.exists('data/europe_energy.csv'):
        df = pd.read_csv('data/europe_energy.csv')
        print(f"✅ Your dataset loaded: {df.shape}")
        return df
    else:
        print("❌ Please place your europe_energy.csv in data/ folder")
        return None

def calculate_real_improvement(df):
    """Calculate real improvement percentage using your data"""
    
    # Use Germany's actual load as target
    target_col = 'DE_load_actual_entsoe_transparency'
    
    if target_col not in df.columns:
        print(f"❌ Column {target_col} not found in your data")
        print(f"Available columns: {[col for col in df.columns if 'DE' in col][:10]}...")
        return None
    
    # Prepare data
    df = df.copy()
    if 'utc_timestamp' in df.columns:
        df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
        df.set_index('utc_timestamp', inplace=True)
    
    # Remove rows with missing target
    df = df[df[target_col].notna()]
    
    print(f"📊 Working with {len(df)} records")
    
    # 1. BASELINE MODEL: Use previous day's value
    df_sorted = df.sort_index()
    baseline_predictions = df_sorted[target_col].shift(24)  # 24 hours ago
    
    # Calculate baseline accuracy
    valid_baseline = baseline_predictions.notna() & df_sorted[target_col].notna()
    y_true_baseline = df_sorted[target_col][valid_baseline]
    y_pred_baseline = baseline_predictions[valid_baseline]
    
    baseline_mae = mean_absolute_error(y_true_baseline, y_pred_baseline)
    print(f"📈 Baseline MAE (previous day): {baseline_mae:.2f} MW")
    
    # 2. ADVANCED MODEL: Prepare features
    features = []
    
    # Lag features
    for lag in [1, 2, 3, 24, 48, 168]:  # 1h, 2h, 3h, 1d, 2d, 1 week
        lag_col = f'lag_{lag}'
        df[lag_col] = df[target_col].shift(lag)
        features.append(lag_col)
    
    # Time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    features.extend(['hour_sin', 'hour_cos', 'day_of_week', 'month'])
    
    # Add other countries' data if available
    for country in ['FR', 'IT', 'ES', 'NL']:
        load_col = f'{country}_load_actual_entsoe_transparency'
        if load_col in df.columns:
            df[f'{country}_load'] = df[load_col]
            features.append(f'{country}_load')
    
    # Remove rows with missing features
    df_clean = df[features + [target_col]].dropna()
    
    if len(df_clean) == 0:
        print("❌ No data after cleaning!")
        return None
    
    X = df_clean[features]
    y = df_clean[target_col]
    
    # Split data (chronological split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Train advanced model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_advanced = model.predict(X_test)
    
    # Calculate advanced model accuracy
    advanced_mae = mean_absolute_error(y_test, y_pred_advanced)
    print(f"🚀 Advanced Model MAE: {advanced_mae:.2f} MW")
    
    # 3. CALCULATE IMPROVEMENT
    improvement = ((baseline_mae - advanced_mae) / baseline_mae) * 100
    
    print(f"\n🎯 REAL PERFORMANCE IMPROVEMENT:")
    print(f"   Baseline MAE: {baseline_mae:.2f} MW")
    print(f"   Advanced MAE: {advanced_mae:.2f} MW")
    print(f"   IMPROVEMENT: {improvement:+.1f}%")
    
    return improvement

def main():
    print("🎯 Calculating REAL Performance from Your Data")
    print("=" * 50)
    
    # Load your actual data
    df = load_your_data()
    if df is None:
        return
    
    # Calculate real improvement
    improvement = calculate_real_improvement(df)
    
    if improvement is not None:
        print(f"\n✅ You can use {improvement:.1f}% improvement in your CV!")
        print("   This is calculated from YOUR actual data")
    else:
        print("\n❌ Could not calculate improvement")

if __name__ == "__main__":
    main()
