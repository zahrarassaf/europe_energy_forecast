import os
import sys

sys.path.append('src')

def main():
    print("🎯 European Energy Forecasting - REAL Calculation")
    print("=" * 60)
    print("📁 Using YOUR Google Drive dataset")
    print("=" * 60)
    
    try:
        from data_collection.data_loader import download_real_dataset, manual_download_instructions
        from models.real_improvement_calculator import RealImprovementCalculator
        
        # 1. Load your REAL dataset
        print("1. Loading your dataset from Google Drive...")
        df = download_real_dataset()
        
        if df is None:
            print("❌ Automated download failed")
            manual_download_instructions()
            return
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # 2. Calculate REAL improvement
        print("\n2. Calculating REAL improvement from your data...")
        calculator = RealImprovementCalculator()
        
        improvement = calculator.calculate_real_improvement(df)
        
        if improvement is None:
            print("❌ Could not calculate improvement")
            return
        
        # 3. Show REAL results
        results = calculator.get_detailed_results()
        print(f"\n" + "=" * 50)
        print(f"🎯 REAL RESULTS FROM YOUR DATA:")
        print(f"   Baseline MAE: {results['baseline_mae']:.2f}")
        print(f"   Advanced MAE: {results['advanced_mae']:.2f}")
        print(f"   Improvement: {results['improvement_percentage']:+.1f}%")
        print(f"   ✅ This is a REAL calculation!")
        print("=" * 50)
        
        print(f"\n📄 You can now use {improvement:.1f}% in your CV and research papers!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
