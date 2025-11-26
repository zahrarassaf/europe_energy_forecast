import pandas as pd
import numpy as np
import os
import sys

print("🎯 European Energy Forecasting - PhD Project")
print("=" * 50)

def download_dataset():
    """Download dataset from Google Drive if not exists"""
    data_path = "data/europe_energy.csv"
    
    if os.path.exists(data_path):
        print("✅ Dataset already exists")
        return pd.read_csv(data_path)
    
    print("📥 Downloading dataset from Google Drive...")
    
    try:
        import gdown
        
        # Google Drive file ID from your link
        file_id = "1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Create data directory if not exists
        os.makedirs("data", exist_ok=True)
        
        # Download file
        gdown.download(url, data_path, quiet=False)
        print("✅ Dataset downloaded successfully!")
        
        return pd.read_csv(data_path)
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please install gdown: pip install gdown")
        return None

def main():
    # Configuration
    COUNTRIES = ['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE', 'PL']
    TARGET_COUNTRY = 'DE'
    
    # 1. Download or load dataset
    df = download_dataset()
    
    if df is None:
        print("🚨 Could not load dataset. Exiting.")
        return
    
    # 2. Show data info
    print(f"\n📊 Data Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Convert DateTime if exists
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        print(f"   Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    
    # 3. Basic analysis
    print(f"\n📈 Statistics for {TARGET_COUNTRY}:")
    if TARGET_COUNTRY in df.columns:
        target_data = df[TARGET_COUNTRY]
        print(f"   Records: {len(target_data):,}")
        print(f"   Mean: {target_data.mean():.2f}")
        print(f"   Std:  {target_data.std():.2f}")
        print(f"   Min:  {target_data.min():.2f}")
        print(f"   Max:  {target_data.max():.2f}")
    
    # 4. Show sample of data
    print(f"\n👀 Sample of data:")
    print(df.head())
    
    print("\n🎉 Project is ready! You can now run advanced models.")
    print("💾 Dataset is automatically downloaded from Google Drive")

if __name__ == "__main__":
    main()
