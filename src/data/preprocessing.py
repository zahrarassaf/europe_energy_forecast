import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class EnergyDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.countries = ['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE']
    
    def load_data(self, file_path='data/time_series.csv'):
        """Load and basic preprocessing from your notebook"""
        df = pd.read_csv(file_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        return df
    
    def create_features(self, df):
        """Create advanced features from your EDA"""
        # Temporal features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Lag features
        for country in self.countries:
            df[f'{country}_lag_7'] = df[country].shift(7)
            df[f'{country}_lag_30'] = df[country].shift(30)
        
        return df.dropna()
    
    def prepare_train_test(self, df, target_country='DE', test_size=0.2):
        """Prepare data for modeling"""
        features = [col for col in df.columns if col not in self.countries]
        X = df[features]
        y = df[target_country]
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
