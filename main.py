import pandas as pd
import os
from download_from_drive import download_real_dataset

def main():
    print("🎯 European Energy Forecasting - REAL Dataset")
    print("=" * 50)
    
    # Load your REAL dataset
    df = download_real_dataset()
    
    if df is not None:
        print(f"✅ Using your REAL dataset: {df.shape}")
        
        # Calculate REAL improvement
        improvement = calculate_real_improvement(df)
        
        if improvement:
            print(f"🎯 REAL RESULT: {improvement:.1f}% improvement")
        else:
            print("❌ Could not calculate improvement")
    else:
        print("🚨 Please ensure your Google Drive file is accessible")

if __name__ == "__main__":
    main()
