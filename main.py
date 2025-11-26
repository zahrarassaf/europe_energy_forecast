import pandas as pd
import numpy as np
import os
import sys

print("🎯 European Energy Forecasting - PhD Project")
print("=" * 50)

def load_and_analyze_data():
    """Load and analyze the new dataset"""
    data_path = "data/europe_energy.csv"
    
    if not os.path.exists(data_path):
        print("❌ Dataset not found!")
        return None
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Convert timestamp
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df.set_index('utc_timestamp', inplace=True)
    
    return df

def main():
    # Load data
    df = load_and_analyze_data()
    if df is None:
        return
    
    # Configuration for new dataset
    TARGET_COLUMN = 'DE_load_actual_entsoe_transparency'
    COUNTRIES = ['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE', 'PL']
    
    # 1. Data overview
    print(f"\n📊 Dataset Overview:")
    print(f"   Time range: {df.index.min()} to {df.index.max()}")
    print(f"   Frequency: Hourly data")
    print(f"   Total hours: {len(df):,}")
    print(f"   Total countries/regions: {len([col for col in df.columns if 'load_actual' in col])}")
    
    # 2. Target analysis
    print(f"\n🎯 Target Analysis ({TARGET_COLUMN}):")
    if TARGET_COLUMN in df.columns:
        target_data = df[TARGET_COLUMN].dropna()
        print(f"   Available records: {len(target_data):,}")
        print(f"   Missing values: {df[TARGET_COLUMN].isnull().sum()}")
        print(f"   Mean: {target_data.mean():.2f} MW")
        print(f"   Std:  {target_data.std():.2f} MW")
        print(f"   Min:  {target_data.min():.2f} MW") 
        print(f"   Max:  {target_data.max():.2f} MW")
    
    # 3. Show available countries
    print(f"\n🌍 Available Country Data:")
    for country in COUNTRIES:
        load_col = f'{country}_load_actual_entsoe_transparency'
        if load_col in df.columns:
            available = df[load_col].notna().sum()
            print(f"   {country}: {available:,} records")
    
    # 4. Feature categories
    print(f"\n📈 Data Categories Available:")
    categories = {
        'Load Actual': [col for col in df.columns if 'load_actual' in col],
        'Load Forecast': [col for col in df.columns if 'load_forecast' in col],
        'Solar Generation': [col for col in df.columns if 'solar_generation' in col],
        'Wind Generation': [col for col in df.columns if 'wind_generation' in col and 'capacity' not in col],
        'Price Data': [col for col in df.columns if 'price_day_ahead' in col]
    }
    
    for category, columns in categories.items():
        print(f"   {category}: {len(columns)} columns")
    
    print(f"\n💡 This dataset is PERFECT for PhD research!")
    print(f"   - Real-time energy data from ENTSO-E")
    print(f"   - Multiple countries and regions") 
    print(f"   - Rich features for advanced modeling")
    print(f"   - Hourly frequency for precise forecasting")

if __name__ == "__main__":
    main()
