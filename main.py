import pandas as pd
import numpy as np
import os
import sys

sys.path.append('src')

from data_collection.data_loader import EuropeanEnergyDataLoader
from eda_statistical.statistical_tests import AdvancedStatisticalAnalyzer
from feature_engineering.advanced_features import AdvancedFeatureEngineer
from advanced_modeling.transformer_models import AdvancedEnergyPredictor
from advanced_modeling.hybrid_models import HybridEnergyForecaster

def main():
    print("=== European Energy Forecasting System ===")
    
    # Step 1: Data Collection
    print("\n1. Loading data...")
    data_loader = EuropeanEnergyDataLoader()
    master_data = data_loader.create_master_dataset()
    data_loader.save_datasets()
    
    # Step 2: Statistical Analysis
    print("\n2. Performing statistical analysis...")
    analyzer = AdvancedStatisticalAnalyzer(master_data)
    statistical_report = analyzer.generate_statistical_report()
    
    # Step 3: Feature Engineering
    print("\n3. Creating features...")
    feature_engineer = AdvancedFeatureEngineer()
    features_data = feature_engineer.create_complete_feature_set(master_data.reset_index())
    
    # Step 4: Modeling
    print("\n4. Training models...")
    
    # Transformer Model
    print("\n4.1 Training Transformer Model...")
    transformer_model = AdvancedEnergyPredictor(sequence_length=30)
    train_loader, test_loader = transformer_model.prepare_data(features_data)
    transformer_losses = transformer_model.train(epochs=50)
    
    # Hybrid Model
    print("\n4.2 Training Hybrid Model...")
    X = features_data.drop(columns=['energy_consumption_mwh']).values
    y = features_data['energy_consumption_mwh'].values
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    hybrid_model = HybridEnergyForecaster()
    hybrid_model.train_base_models(X_train, y_train, X_test, y_test)
    hybrid_predictions = hybrid_model.train_meta_model(X_train, y_train, X_test, y_test)
    
    print("\n=== Model Training Complete ===")
    print("Models available: Transformer, Hybrid Ensemble")
    print("Feature engineering completed")
    print("Statistical analysis report generated")
    
    return {
        'data': master_data,
        'features': features_data,
        'transformer_model': transformer_model,
        'hybrid_model': hybrid_model,
        'statistical_report': statistical_report
    }

if __name__ == "__main__":
    results = main()
