import pandas as pd
import numpy as np
from config.research_config import config
from src.data.preprocessing import EnergyDataPreprocessor
from src.models.transformer_model import EnergyTransformer
import torch
import torch.optim as optim
import os

def check_environment():
    """Check if all requirements are met"""
    print("=== ENVIRONMENT CHECK ===")
    
    # Check data file
    if not os.path.exists(config.DATA_PATH):
        print(f"❌ CRITICAL: Data file not found at {config.DATA_PATH}")
        print("Please make sure 'data/europe_energy.csv' exists in your repository")
        return False
    
    print(f"✅ Data file found: {config.DATA_PATH}")
    
    # Check required directories
    required_dirs = ['src/data', 'src/models', 'config']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Directory missing: {dir_path}")
            return False
    
    print("✅ All directories exist")
    return True

def main():
    print("=== PhD-Level European Energy Forecasting ===")
    
    # Environment check
    if not check_environment():
        print("\n🚨 Please fix the above issues before continuing")
        return
    
    print("\n1. Loading and preprocessing data...")
    preprocessor = EnergyDataPreprocessor(config)
    
    try:
        df = preprocessor.load_data()
        print(f"Data columns: {list(df.columns)}")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print("\n2. Creating advanced features...")
    try:
        df_processed = preprocessor.create_advanced_features(df)
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Available features: {[col for col in df_processed.columns if col not in config.COUNTRIES][:10]}...")
        
    except Exception as e:
        print(f"❌ Error creating features: {e}")
        return
    
    print("\n3. Preparing sequences for model training...")
    try:
        X_sequences, y_sequences = preprocessor.prepare_sequences(df_processed, config.TARGET_COUNTRY)
        print(f"X_sequences shape: {X_sequences.shape}")
        print(f"y_sequences shape: {y_sequences.shape}")
        
    except Exception as e:
        print(f"❌ Error preparing sequences: {e}")
        return
    
    # Check if we have enough data
    if len(X_sequences) < 100:
        print(f"⚠️ Warning: Only {len(X_sequences)} sequences available")
        print("Consider reducing SEQUENCE_LENGTH in config")
    
    print("\n4. Setting up Transformer model...")
    try:
        input_dim = X_sequences.shape[2]
        model = EnergyTransformer(input_dim, config.SEQUENCE_LENGTH)
        print(f"✅ Transformer model created with input_dim={input_dim}")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    print("\n5. Demo training...")
    try:
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_sequences[:100])  # Use subset for demo
        y_tensor = torch.FloatTensor(y_sequences[:100])
        
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
            print(f"   Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        
        print("✅ Demo training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in training: {e}")
        return
    
    print("\n🎉 PROJECT IS WORKING CORRECTLY!")
    print("Next: Implement full training, validation, and evaluation")

if __name__ == "__main__":
    main()
