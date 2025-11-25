import pandas as pd
import numpy as np
from config.research_config import config
from src.data.preprocessing import EnergyDataPreprocessor
from src.models.transformer_model import EnergyTransformer
from src.analysis.statistical_tests import AdvancedStatisticalAnalyzer
from src.utils.visualization import ResearchVisualizer
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_statistical_analysis(df):
    """Run comprehensive statistical analysis"""
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS")
    print("="*50)
    
    analyzer = AdvancedStatisticalAnalyzer(df)
    
    # Stationarity tests for all countries
    for country in config.COUNTRIES:
        stationarity = analyzer.comprehensive_stationarity_test(country)
        print(f"{country}: ADF p-value = {stationarity['adf']['p_value']:.4f}, "
              f"Stationary = {stationarity['is_stationary']}")

def train_transformer_model(X_train, y_train, X_test, y_test, input_dim):
    """Train and evaluate Transformer model"""
    print("\n" + "="*50)
    print("TRANSFORMER MODEL TRAINING")
    print("="*50)
    
    model = EnergyTransformer(input_dim, config.SEQUENCE_LENGTH)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop (simplified)
    model.train()
    for epoch in range(5):  # Demo - use config.EPOCHS in real scenario
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions.squeeze(), y_test)
        
    print(f"Test Loss: {test_loss.item():.4f}")
    return model, predictions

def main():
    print("=== PhD-Level European Energy Forecasting ===")
    
    # 1. Data preprocessing
    print("\n1. Loading and preprocessing data...")
    preprocessor = EnergyDataPreprocessor(config)
    df = preprocessor.load_data()
    df_processed = preprocessor.create_advanced_features(df)
    
    print(f"Original data shape: {df.shape}")
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Number of features: {len([col for col in df_processed.columns if col not in config.COUNTRIES])}")
    
    # 2. Statistical analysis
    run_statistical_analysis(df)
    
    # 3. Prepare sequences for deep learning
    print("\n2. Preparing sequences for model training...")
    X_sequences, y_sequences = preprocessor.prepare_sequences(df_processed, config.TARGET_COUNTRY)
    
    print(f"Sequences shape: {X_sequences.shape}")
    print(f"Target shape: {y_sequences.shape}")
    
    # 4. Train-test split
    split_idx = int(len(X_sequences) * (1 - config.TEST_SIZE))
    X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
    y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # 5. Model training
    input_dim = X_sequences.shape[2]
    model, predictions = train_transformer_model(
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim
    )
    
    print("\n" + "="*50)
    print("PROJECT SETUP COMPLETE")
    print("="*50)
    print("Next steps:")
    print("1. Implement full training loop in transformer_model.py")
    print("2. Add more advanced models (LSTM, Prophet, XGBoost)")
    print("3. Implement cross-validation")
    print("4. Add hyperparameter optimization")
    print("5. Create publication-ready visualizations")

if __name__ == "__main__":
    main()
