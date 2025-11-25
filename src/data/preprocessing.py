import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnergyDataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load data from your CSV file"""
        df = pd.read_csv(self.config.DATA_PATH)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        return df
    
    def create_advanced_features(self, df):
        """Create PhD-level features based on your EDA"""
        
        # Temporal features from your notebook
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['day_of_week'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['week_of_year'] = df.index.isocalendar().week
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Lag features for all countries
        for country in self.config.COUNTRIES:
            for lag in self.config.LAGS:
                df[f'{country}_lag_{lag}'] = df[country].shift(lag)
            
            # Rolling statistics
            for window in self.config.ROLLING_WINDOWS:
                df[f'{country}_rolling_mean_{window}'] = df[country].rolling(window).mean()
                df[f'{country}_rolling_std_{window}'] = df[country].rolling(window).std()
        
        # Cross-country correlations
        for i, country1 in enumerate(self.config.COUNTRIES):
            for country2 in self.config.COUNTRIES[i+1:]:
                df[f'corr_{country1}_{country2}'] = df[country1].rolling(30).corr(df[country2])
        
        return df.dropna()
    
    def prepare_sequences(self, df, target_country):
        """Prepare sequences for LSTM/Transformer models"""
        features = [col for col in df.columns if col not in self.config.COUNTRIES]
        target = df[target_country]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(df[features])
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_scaled) - self.config.SEQUENCE_LENGTH):
            X_sequences.append(X_scaled[i:i + self.config.SEQUENCE_LENGTH])
            y_sequences.append(target.iloc[i + self.config.SEQUENCE_LENGTH])
        
        return np.array(X_sequences), np.array(y_sequences)
