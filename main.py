from src.data.preprocessing import EnergyDataPreprocessor
from src.analysis.statistical_tests import AdvancedStatisticalTests
from src.models.advanced_models import AdvancedEnergyModels

def main():
    print("=== European Energy Forecasting - PhD Level ===")
    
    # 1. Data preprocessing
    print("1. Loading and preprocessing data...")
    preprocessor = EnergyDataPreprocessor()
    df = preprocessor.load_data()
    df_processed = preprocessor.create_features(df)
    
    # 2. Statistical analysis
    print("2. Performing statistical tests...")
    analyzer = AdvancedStatisticalTests(df)
    for country in preprocessor.countries:
        stationarity = analyzer.stationarity_analysis(country)
        print(f"{country} - Stationary: {stationarity['is_stationary']}")
    
    # 3. Prepare data for modeling
    print("3. Preparing training data...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(df_processed)
    
    # 4. Train advanced models
    print("4. Training advanced models...")
    model_trainer = AdvancedEnergyModels()
    model_trainer.create_ensemble()
    results = model_trainer.evaluate_models(X_train, X_test, y_train, y_test)
    
    # 5. Display results
    print("\n=== Model Results ===")
    for model_name, metrics in results.items():
        print(f"{model_name}: RMSE = {metrics['rmse']:.2f}, MAE = {metrics['mae']:.2f}")

if __name__ == "__main__":
    main()
