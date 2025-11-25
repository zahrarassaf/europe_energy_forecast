import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_names = []
        self.scalers = {}
        
    def create_comprehensive_temporal_features(self, df, date_column='date'):
        if date_column in df.columns:
            df = df.set_index(date_column)
        
        df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        df['week_of_year_sin'] = np.sin(2 * np.pi * df.index.isocalendar().week / 52)
        df['week_of_year_cos'] = np.cos(2 * np.pi * df.index.isocalendar().week / 52)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        df['is_weekend'] = df.index.dayofweek >= 5
        df['is_holiday_season'] = df.index.month.isin([12, 1])
        df['is_summer'] = df.index.month.isin([6, 7, 8])
        df['is_winter'] = df.index.month.isin([12, 1, 2])
        
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        df['is_year_start'] = df.index.is_year_start.astype(int)
        df['is_year_end'] = df.index.is_year_end.astype(int)
        
        return df
    
    def create_sophisticated_lag_features(self, df, target_col, lags=[1, 2, 3, 7, 14, 30, 365]):
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        windows = [7, 14, 30, 90]
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            df[f'{target_col}_rolling_median_{window}'] = df[target_col].rolling(window=window).median()
            
            df[f'{target_col}_rolling_change_{window}'] = df[target_col] / df[f'{target_col}_rolling_mean_{window}']
        
        for alpha in [0.1, 0.3, 0.5]:
            df[f'{target_col}_ewm_{alpha}'] = df[target_col].ewm(alpha=alpha).mean()
        
        return df
    
    def create_interaction_features(self, df):
        if 'temperature_c' in df.columns and 'month_sin' in df.columns:
            df['temp_season_interaction'] = df['temperature_c'] * df['month_sin']
        
        df['holiday_weekend_interaction'] = df['is_holiday_season'] * df['is_weekend']
        
        if 'gdp_growth_pct' in df.columns and 'energy_consumption_mwh_lag_30' in df.columns:
            df['gdp_energy_interaction'] = df['gdp_growth_pct'] * df['energy_consumption_mwh_lag_30']
        
        return df
    
    def create_advanced_statistical_features(self, df, target_col):
        df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1)
        df[f'{target_col}_pct_change_7'] = df[target_col].pct_change(7)
        df[f'{target_col}_pct_change_30'] = df[target_col].pct_change(30)
        
        df[f'{target_col}_cumulative_mean'] = df[target_col].expanding().mean()
        df[f'{target_col}_cumulative_std'] = df[target_col].expanding().std()
        
        df[f'{target_col}_volatility_30'] = df[target_col].pct_change().rolling(30).std()
        
        return df
    
    def create_fourier_features(self, df, periods=[365, 182, 90]):
        for period in periods:
            for i in range(1, 4):
                df[f'fourier_sin_{period}_{i}'] = np.sin(2 * np.pi * i * np.arange(len(df)) / period)
                df[f'fourier_cos_{period}_{i}'] = np.cos(2 * np.pi * i * np.arange(len(df)) / period)
        
        return df
    
    def feature_selection(self, df, target_col, k=20):
        df_clean = df.dropna()
        
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        selector_mutual = SelectKBest(score_func=mutual_info_regression, k=k)
        selector_f = SelectKBest(score_func=f_regression, k=k)
        
        X_selected_mutual = selector_mutual.fit_transform(X, y)
        X_selected_f = selector_f.fit_transform(X, y)
        
        mutual_features = X.columns[selector_mutual.get_support()].tolist()
        f_features = X.columns[selector_f.get_support()].tolist()
        
        selected_features = list(set(mutual_features + f_features))
        
        print(f"Selected {len(selected_features)} features")
        return selected_features
    
    def create_complete_feature_set(self, df, target_col='energy_consumption_mwh'):
        print("Creating comprehensive feature set...")
        
        df = self.create_comprehensive_temporal_features(df)
        df = self.create_sophisticated_lag_features(df, target_col)
        df = self.create_interaction_features(df)
        df = self.create_advanced_statistical_features(df, target_col)
        df = self.create_fourier_features(df)
        
        selected_features = self.feature_selection(df, target_col)
        
        final_features = selected_features + [target_col]
        df_final = df[final_features].copy()
        
        print(f"Final dataset shape: {df_final.shape}")
        return df_final

if __name__ == "__main__":
    dates = pd.date_range('2015-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'date': dates,
        'energy_consumption_mwh': 100000 + 5000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + 
                             100 * np.arange(len(dates)) / 365 + np.random.normal(0, 1000, len(dates)),
        'temperature_c': 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 3, len(dates)),
        'gdp_growth_pct': np.random.normal(2, 0.5, len(dates))
    })
    
    engineer = AdvancedFeatureEngineer()
    final_data = engineer.create_complete_feature_set(sample_data)
