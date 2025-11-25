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
        """Load data with proper error handling"""
        try:
            df = pd.read_csv(self.config.DATA_PATH)
            print(f"✅ Data loaded successfully: {df.shape}")
            
            # Check if required columns exist
            required_cols = ['DateTime'] + self.config.COUNTRIES
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
                
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df.set_index('DateTime', inplace=True)
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at: {self.config.DATA_PATH}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    def create_advanced_features(self, df):
        """Create features with safe week calculation"""
        print("Creating advanced features...")
        
        # Basic temporal features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['day_of_week'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        
        # Safe week calculation
        try:
            df['week_of_year'] = df.index.isocalendar().week
        except:
            # Fallback method
            df['week_of_year'] = (df.index.dayofyear / 7).astype(int)
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Features only for target country to avoid over-complexity
        target_country = self.config.TARGET_COUNTRY
        print(f"Creating features for target country: {target_country}")
        
        for lag in [1, 7, 30]:  # Reduced lags for stability
            df[f'{target_country}_lag_{lag}'] = df[target_country].shift(lag)
        
        for window in [7, 30]:  # Reduced windows
            df[f'{target_country}_rolling_mean_{window}'] = df[target_country].rolling(window).mean()
            df[f'{target_country}_rolling_std_{window}'] = df[target_country].rolling(window).std()
        
        result = df.dropna()
        print(f"✅ Features created. Final shape: {result.shape}")
        return result
    
    def prepare_sequences(self, df, target_country):
        """Prepare sequences with proper validation"""
        print("Preparing sequences for training...")
        
        # Select only feature columns (exclude original country data)
        feature_columns = [col for col in df.columns if col not in self.config.COUNTRIES]
        
        if len(feature_columns) == 0:
            raise ValueError("No features found for modeling")
            
        print(f"Using {len(feature_columns)} features")
        
        target = df[target_country]
        X_scaled = self.scaler.fit_transform(df[feature_columns])
        
        X_sequences = []
        y_sequences = []
        
        n_sequences = len(X_scaled) - self.config.SEQUENCE_LENGTH
        if n_sequences <= 0:
            raise ValueError(f"Not enough data for sequence length {self.config.SEQUENCE_LENGTH}")
        
        for i in range(n_sequences):
            X_sequences.append(X_scaled[i:i + self.config.SEQUENCE_LENGTH])
            y_sequences.append(target.iloc[i + self.config.SEQUENCE_LENGTH])
        
        print(f"✅ Created {len(X_sequences)} sequences")
        return np.array(X_sequences), np.array(y_sequences)
