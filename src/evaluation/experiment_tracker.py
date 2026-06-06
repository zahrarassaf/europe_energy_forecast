 ============================================================================
import sys
import os

# Fix for Jupyter: if sys.argv is empty or has invalid values
if len(sys.argv) == 0 or not sys.argv[0]:
    sys.argv = ['']  # Add a dummy argument for ArgumentParser
# ============================================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import gc
import json
import hashlib
import argparse
import yaml
import random
import itertools
import signal
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# 1
# ============================================================================
from scipy import stats
from scipy.stats import wilcoxon, norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import psutil

# ============================================================================
# OPTIONAL IMPORTS WITH GRACEFUL FALLBACK
# ============================================================================
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================
BASE_SEED = 42
N_SEEDS = 5

ALL_COUNTRIES_FULL = [
    "AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES",
    "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IT", "LT", "LU",
    "LV", "ME", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI",
    "SK", "UA"
]

PROBLEM_COUNTRIES = ["SK", "HU", "PT", "PL", "HR", "UA"]

ALL_COUNTRIES = [c for c in ALL_COUNTRIES_FULL if c not in PROBLEM_COUNTRIES]

# ============================================================================
# REPRODUCIBILITY AND DETERMINISM
# ============================================================================

def set_seed(seed, deterministic=True):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Integer seed value
        deterministic: If True, enforce GPU determinism (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.experimental.enable_op_determinism()
    else:
        os.environ['TF_DETERMINISTIC_OPS'] = '0'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '0'

set_seed(BASE_SEED, deterministic=True)

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================
gc.enable()
gc.set_threshold(100, 5, 5)

def check_memory_and_clear(threshold_gb=5.0, force=False):
    """
    Monitor memory usage and trigger garbage collection if needed.
    
    Args:
        threshold_gb: Memory threshold in GB
        force: Force cleanup regardless of memory usage
    
    Returns:
        bool: True if cleanup was performed
    """
    try:
        process = psutil.Process()
        mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
        if mem_gb > threshold_gb or force:
            print(f"Memory usage: {mem_gb:.2f} GB - Cleaning up...")
            tf.keras.backend.clear_session()
            for _ in range(3):
                gc.collect()
            return True
    except:
        pass
    return False


# ============================================================================
# CONFIGURATION MANAGEMENT - FIXED FOR ALL COUNTRIES
# ============================================================================

class Config:
    """
    Configuration management for experiments.
    All parameters are documented with their scientific justification.
    """
    
    @staticmethod
    def load_from_yaml(path):
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def save_to_yaml(config, path):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @staticmethod
    def get_default_config():
        """
        Default configuration with scientific justification for each parameter.
        """
        return {
            # ========== DATA PARAMETERS ==========
            # FIXED: Correct data path format using forward slashes
            "data_path": "C:/Users/Zahara/Documents/Zoom/europe_energy_forecast/data/europe_energy_real.csv",
            "country_code": "AT",  # Default single country
            "target_column_template": "{country}_load_actual_entsoe_transparency",  # Template for target column
            "timestamp_column": "utc_timestamp",  # Column name for timestamp
            "cet_timestamp_column": "cet_cest_timestamp",  # Optional CET timestamp
            
            # ========== MULTI-COUNTRY PARAMETERS ==========
            "run_multi_country": True,  # FIXED: Now runs all countries by default
            "countries": ALL_COUNTRIES,  # List of all countries to process
            "skip_if_exists": True,  # Skip countries that already have results
            "max_countries": None,  # FIXED: None means process ALL available countries
            
            # ========== SEQUENCE PARAMETERS ==========
            "sequence_length": 24,  # 24 hours of history (daily pattern)
            "forecast_horizon": 6,   # 6 hours ahead (operational planning)
            
            # ========== SEQUENCE LENGTH SENSITIVITY ==========
            "test_sequence_lengths": [24, 48, 72, 168],  # Test multiple sequence lengths
            "best_sequence_length": 24,  # Will be determined by testing
            
            # ========== DATA SPLIT PARAMETERS ==========
            "test_size": 0.2,        # 20% for final evaluation
            "val_size": 0.1,          # 10% for validation (from training data)
            
            # ========== MODEL ARCHITECTURE ==========
            "model_type": "simple_lstm",
            # Smaller model to prevent overfitting
            "lstm_units_1": 32,
            "lstm_units_2": 16,
            "dense_units_1": 16,
            "dense_units_2": 8,
            "dropout_rate": 0.3,
            "recurrent_dropout": 0.1,
            "l2_regularization": 0.001,
            
            # ========== TRAINING PARAMETERS ==========
            "learning_rate": 0.0003,
            "batch_size": 32,
            "epochs": 200,
            "patience": 5,
            "min_delta": 0.001,
            "gradient_clip": 1.0,
            
            # ========== FEATURE ENGINEERING ==========
            "use_features": True,
            "n_fourier_terms": 4,
            "fourier_periods": [24, 168],  # Daily and weekly seasonality
            
            # ========== FEATURE SELECTION ==========
            "include_price_features": True,
            "include_solar_features": True,
            "include_wind_features": True,
            "include_forecast_features": True,
            "max_features_per_country": 10,  # Limit features to prevent overfitting
            
            # ========== ABLATION COMPONENTS ==========
            "use_attention": False,
            "use_bidirectional": False,
            "use_residual": False,
            "use_multi_head_attention": False,
            
            # ========== EXPERIMENT PARAMETERS ==========
            "n_seeds": N_SEEDS,
            "optimize_hyperparameters": False,
            "n_trials": 20,
            "n_folds_cv": 3,
            
            # ========== REPRODUCIBILITY ==========
            "seed": BASE_SEED,
            "deterministic": True,
            "seed_list": [BASE_SEED + i for i in range(N_SEEDS)],
            
            # ========== UNCERTAINTY PARAMETERS ==========
            "quantiles": [0.1, 0.5, 0.9],
            "seasonal_period": 24,
            
            # ========== BASELINES ==========
            "include_persistence_baseline": True,
        }


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def compute_dataset_hash(df):
    """Compute SHA256 hash of dataset for reproducibility tracking."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def create_sequences_vectorized(data, seq_len, horizon):
    """
    Create input-output sequences for time series forecasting.
    
    IMPORTANT: This function assumes the target is the first column.
    For multivariate forecasting, ensure data is properly formatted.
    
    Args:
        data: numpy array of shape (n_timesteps, n_features)
              The first column is assumed to be the target.
        seq_len: input sequence length
        horizon: forecast horizon
    
    Returns:
        X: input sequences of shape (n_samples, seq_len, n_features)
        y: target sequences of shape (n_samples, horizon)
    """
    n_samples = len(data) - seq_len - horizon + 1
    if n_samples <= 0:
        return None, None
    
    # Create indices for vectorized indexing
    indices = np.arange(n_samples)[:, None] + np.arange(seq_len)
    X = data[indices]
    
    # Target indices - using the first column (target)
    target_indices = np.arange(n_samples)[:, None] + np.arange(seq_len, seq_len + horizon)
    y = data[target_indices, 0]  # Only take the target column
    
    return X, y

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1):
    """
    Diebold-Mariano test for forecast comparison with Newey-West HAC correction.
    
    This implementation uses the correct variance estimator for multi-step forecasts
    as described in Diebold & Mariano (1995) and subsequent corrections.
    
    Args:
        y_true: actual values
        y_pred1: predictions from model 1
        y_pred2: predictions from model 2
        h: forecast horizon (for autocorrelation correction)
    
    Returns:
        dm_stat: DM test statistic
        p_value: p-value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred1 = np.asarray(y_pred1).flatten()
    y_pred2 = np.asarray(y_pred2).flatten()
    
    min_len = min(len(y_true), len(y_pred1), len(y_pred2))
    y_true = y_true[:min_len]
    y_pred1 = y_pred1[:min_len]
    y_pred2 = y_pred2[:min_len]
    
    # Loss differential
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    d = e1**2 - e2**2
    d = d[~np.isnan(d)]
    
    if len(d) < 2:
        return 0.0, 1.0
    
    d_bar = np.mean(d)
    n = len(d)
    
    # Newey-West HAC variance estimator
    # Accounts for autocorrelation in multi-step forecasts
    gamma = np.correlate(d - d_bar, d - d_bar, mode='full')[n-1:]
    gamma = gamma / n
    
    # Truncation lag for Newey-West (typically h-1)
    max_lag = min(h, n - 1)
    omega = gamma[0] + 2 * np.sum([(1 - l/(max_lag+1)) * gamma[l] for l in range(1, max_lag)])
    
    var_d = omega / n
    
    dm_stat = d_bar / np.sqrt(var_d) if var_d > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

def holm_bonferroni_correction(p_values, alpha=0.05):
    """
    Holm-Bonferroni correction for multiple hypothesis testing.
    
    Args:
        p_values: list of p-values
        alpha: significance level
    
    Returns:
        list of booleans indicating which hypotheses are rejected
    """
    n_tests = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    
    reject = np.zeros(n_tests, dtype=bool)
    for i, p in enumerate(sorted_p):
        if p < alpha / (n_tests - i):
            reject[sorted_idx[i]] = True
    
    return reject.tolist()

def moving_block_bootstrap(data, block_size=24, n_bootstrap=1000, ci=0.95):
    """
    Moving block bootstrap for time series data.
    Preserves autocorrelation structure within blocks.
    
    Args:
        data: time series data
        block_size: size of blocks (typically seasonal period)
        n_bootstrap: number of bootstrap samples
        ci: confidence level
    
    Returns:
        lower: lower confidence bound
        upper: upper confidence bound
    """
    n = len(data)
    n_blocks = int(np.ceil(n / block_size))
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        blocks = []
        for _ in range(n_blocks):
            start = np.random.randint(0, max(1, n - block_size))
            blocks.append(data[start:start+block_size])
        bootstrap_sample = np.concatenate(blocks)[:n]
        bootstrap_stats.append(np.mean(bootstrap_sample))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    
    return lower, upper


# ============================================================================
# CENTRALIZED SCALER MANAGER
# ============================================================================

class ScalerManager:
    """
    Centralized scaler manager for consistent scaling across all splits.
    
    SCIENTIFIC DESIGN:
    - All scalers are fitted ONLY on training data
    - Parameters are explicitly stored and reused
    - No information from validation/test leaks into scaling
    """
    
    def __init__(self):
        self.target_scaler = None
        self.feature_scalers = {}
        self.is_fitted = False
    
    def fit_target(self, target_values):
        """Fit target scaler on training data only."""
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(target_values.reshape(-1, 1))
        return self
    
    def transform_target(self, target_values):
        """Transform target values using fitted scaler."""
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted yet. Call fit_target first.")
        return self.target_scaler.transform(target_values.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target_values):
        """Inverse transform target values to original scale."""
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted yet.")
        return self.target_scaler.inverse_transform(target_values.reshape(-1, 1)).flatten()
    
    def fit_feature(self, feature_name, feature_values):
        """Fit a feature scaler on training data only."""
        scaler = StandardScaler()
        scaler.fit(feature_values.reshape(-1, 1))
        self.feature_scalers[feature_name] = scaler
        return scaler
    
    def transform_feature(self, feature_name, feature_values):
        """Transform feature values using fitted scaler."""
        if feature_name not in self.feature_scalers:
            raise ValueError(f"Scaler for feature {feature_name} not fitted yet.")
        return self.feature_scalers[feature_name].transform(feature_values.reshape(-1, 1)).flatten()
    
    def get_feature_scaler(self, feature_name):
        """Get scaler for a specific feature."""
        return self.feature_scalers.get(feature_name)


# ============================================================================
# FEATURE SELECTOR - NEW for handling many features
# ============================================================================

class FeatureSelector:
    """
    Select relevant features for each country.
    
    Based on column naming patterns in your dataset:
    - Price: {country}_price_day_ahead
    - Solar: {country}_solar_generation_actual
    - Wind: {country}_wind_*_generation_actual
    - Forecast: {country}_load_forecast_entsoe_transparency
    """
    
    def __init__(self, config):
        self.config = config
        self.feature_patterns = {
            'price': ['price_day_ahead', 'price'],
            'solar': ['solar_generation_actual', 'solar_capacity', 'solar_profile'],
            'wind': ['wind_generation_actual', 'wind_offshore', 'wind_onshore', 'wind_capacity'],
            'forecast': ['load_forecast_entsoe_transparency', 'forecast']
        }
    
    def get_features_for_country(self, all_columns, country_code):
        """
        Get feature columns for a specific country.
        
        Args:
            all_columns: list of all column names in dataset
            country_code: country code (e.g., 'AT')
        
        Returns:
            list: feature columns for this country
        """
        features = []
        prefix = f"{country_code}_"
        
        # Target column (excluded from features)
        target = f"{country_code}_load_actual_entsoe_transparency"
        
        # Find all columns starting with country code
        for col in all_columns:
            if col.startswith(prefix) and col != target:
                # Check if we should include this feature type
                include = self._should_include_feature(col)
                if include:
                    features.append(col)
        
        # Limit number of features if specified
        max_features = self.config.get('max_features_per_country', 10)
        if len(features) > max_features:
            # Prioritize features by type: forecast > price > solar > wind
            priority_features = self._prioritize_features(features)
            features = priority_features[:max_features]
        
        return features
    
    def _should_include_feature(self, column):
        """Check if feature should be included based on config."""
        if not self.config.get('use_features', True):
            return False
        
        # Check each feature type
        for feature_type, patterns in self.feature_patterns.items():
            config_key = f"include_{feature_type}_features"
            if self.config.get(config_key, True):
                for pattern in patterns:
                    if pattern in column:
                        return True
        
        return False
    
    def _prioritize_features(self, features):
        """Prioritize features by importance."""
        priority = []
        
        # Priority 1: Forecast
        forecast_features = [f for f in features if 'forecast' in f]
        priority.extend(sorted(forecast_features))
        
        # Priority 2: Price
        price_features = [f for f in features if 'price' in f]
        priority.extend(sorted(price_features))
        
        # Priority 3: Solar
        solar_features = [f for f in features if 'solar' in f]
        priority.extend(sorted(solar_features))
        
        # Priority 4: Wind
        wind_features = [f for f in features if 'wind' in f]
        priority.extend(sorted(wind_features))
        
        # Add any remaining features
        remaining = [f for f in features if f not in priority]
        priority.extend(sorted(remaining))
        
        return priority


# ============================================================================
# LEAKAGE-FREE DATA PROCESSOR - FIXED FOR REAL DATA
# ============================================================================

class DataProcessor:
    """
    Data processor with strict leakage prevention.
    
    SCIENTIFIC DESIGN:
    1. All transformations are fitted ONLY on training data
    2. Scaling parameters are saved and reused via ScalerManager
    3. No information from validation/test is used in preprocessing
    4. Temporal alignment is preserved for time series
    5. FIXED: Fourier features use correct angular frequency (2π * k * t / period)
    """
    
    def __init__(self, config):
        self.config = config
        self.data = None
        self.scaler_manager = ScalerManager()
        self.fourier_params = None
        self.dataset_hash = None
        self.feature_stats = {}
        self.target_column = None
        self.feature_selector = FeatureSelector(config)
        self.timestamp_column = config.get('timestamp_column', 'utc_timestamp')
        self.cet_timestamp_column = config.get('cet_timestamp_column', 'cet_cest_timestamp')
        
    def load_data(self, data_path=None):
        """
        Load data from CSV or create sample dataset.
        
        Args:
            data_path: path to CSV file
        
        Returns:
            pandas DataFrame
        """
        if data_path and os.path.exists(data_path):
            print(f"Loading data from: {data_path}")
            self.data = pd.read_csv(data_path, low_memory=False)
            
            # Handle timestamp columns
            if self.timestamp_column in self.data.columns:
                self.data[self.timestamp_column] = pd.to_datetime(self.data[self.timestamp_column])
                self.data.set_index(self.timestamp_column, inplace=True)
            elif self.cet_timestamp_column in self.data.columns:
                self.data[self.cet_timestamp_column] = pd.to_datetime(self.data[self.cet_timestamp_column])
                self.data.set_index(self.cet_timestamp_column, inplace=True)
            else:
                # If no timestamp column, create a default index
                print("  Warning: No timestamp column found. Using default index.")
                self.data.index = pd.date_range('2015-01-01', periods=len(self.data), freq='h')
        else:
            print("Data file not found. Creating sample dataset...")
            self.data = self._create_sample_dataset(20000)
        
        self.dataset_hash = compute_dataset_hash(self.data)
        print(f"Dataset hash: {self.dataset_hash}")
        print(f"Data shape: {self.data.shape}")
        print(f"Number of columns: {len(self.data.columns)}")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        
        # Print available countries
        self._print_available_countries()
        
        return self.data
    
    def _print_available_countries(self):
        """Print all available countries in the dataset."""
        target_cols = [col for col in self.data.columns if '_load_actual_entsoe_transparency' in col]
        countries = sorted(list(set([col.split('_')[0] for col in target_cols])))
        print(f"Available countries: {', '.join(countries)}")
    
    def _create_sample_dataset(self, n_samples):
        """Create synthetic dataset for testing."""
        np.random.seed(BASE_SEED)
        dates = pd.date_range('2015-01-01', periods=n_samples, freq='h')
        
        t = np.arange(n_samples)
        
        yearly = 5000 + 1000 * np.sin(2 * np.pi * t / (24 * 365))
        weekly = 300 * np.sin(2 * np.pi * t / (24 * 7))
        daily = 500 * np.sin(2 * np.pi * t / 24)
        trend = 0.001 * t
        noise_scale = 200 * (1 + 0.5 * np.sin(2 * np.pi * t / (24 * 30)))
        noise = np.random.normal(0, noise_scale, n_samples)
        
        load = yearly + daily + weekly + trend + noise
        
        df = pd.DataFrame({
            'utc_timestamp': dates,
            'AT_load_actual_entsoe_transparency': load,
            'AT_price_day_ahead': 50 + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 5, n_samples),
            'AT_solar_generation_actual': 1000 * np.maximum(0, np.sin(2 * np.pi * (t % 24) / 24)) + np.random.normal(0, 50, n_samples),
            'AT_wind_onshore_generation_actual': 500 * (1 + 0.5 * np.random.randn(n_samples)),
            'DE_load_actual_entsoe_transparency': load * 5 + np.random.normal(0, 1000, n_samples),
            'FR_load_actual_entsoe_transparency': load * 4 + np.random.normal(0, 800, n_samples),
        })
        df.set_index('utc_timestamp', inplace=True)
        
        return df
    
    def set_target_column(self, country_code):
        """Set target column based on country code."""
        template = self.config.get('target_column_template', "{country}_load_actual_entsoe_transparency")
        self.target_column = template.format(country=country_code)
        return self.target_column
    
    def get_feature_columns(self, country_code):
        """
        Get feature columns for a specific country using intelligent selection.
        """
        if self.data is None:
            return []
        
        all_columns = list(self.data.columns)
        return self.feature_selector.get_features_for_country(all_columns, country_code)
    
    def check_country_availability(self, country_code):
        """Check if data for a country is available."""
        target_col = self.set_target_column(country_code)
        
        if target_col not in self.data.columns:
            print(f"  {country_code}: Target column {target_col} not found")
            return False
        
        # Check if enough non-NaN data
        target_data = self.data[target_col].dropna()
        min_required = self.config['sequence_length'] + self.config['forecast_horizon'] + 100
        
        if len(target_data) < min_required:
            print(f"  {country_code}: Not enough data ({len(target_data)} < {min_required})")
            return False
        
        return True
    
    def prepare_splits(self, country_code):
        """
        Prepare train/val/test splits for a specific country with NO LEAKAGE.
        
        CRITICAL: All transformations are fitted on training data only.
        """
        self.set_target_column(country_code)
        
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column {self.target_column} not found for country {country_code}")
        
        # Get feature columns using intelligent selection
        feature_cols = self.get_feature_columns(country_code)
        
        print(f"\nPreparing splits for {country_code} - target: {self.target_column}")
        print(f"  Selected {len(feature_cols)} features:")
        for i, col in enumerate(feature_cols[:5]):  # Show first 5
            print(f"    - {col}")
        if len(feature_cols) > 5:
            print(f"    ... and {len(feature_cols) - 5} more")
        
        # Get raw data
        columns = [self.target_column] + feature_cols
        raw_df = self.data[columns].copy()
        timestamps = self.data.index.copy()
        
        # Calculate split indices (chronological order preserved)
        total_len = len(raw_df)
        test_size = self.config.get('test_size', 0.2)
        val_size = self.config.get('val_size', 0.1)
        
        train_end = int(total_len * (1 - test_size - val_size))
        val_end = train_end + int(total_len * val_size)
        
        # Create raw splits
        train_raw_df = raw_df.iloc[:train_end].copy()
        val_raw_df = raw_df.iloc[train_end:val_end].copy()
        test_raw_df = raw_df.iloc[val_end:].copy()
        
        train_timestamps = timestamps[:train_end]
        val_timestamps = timestamps[train_end:val_end]
        test_timestamps = timestamps[val_end:]
        
        print(f"  Split sizes - Train: {len(train_raw_df)}, Val: {len(val_raw_df)}, Test: {len(test_raw_df)}")
        
        # ========== HANDLE MISSING VALUES ==========
        # Fit on train only, then apply to val/test
        train_raw_df = self._handle_missing_values(train_raw_df, fit=True)
        val_raw_df = self._handle_missing_values(val_raw_df, fit=False)
        test_raw_df = self._handle_missing_values(test_raw_df, fit=False)
        
        # Extract target
        train_target = train_raw_df[self.target_column].values
        val_target = val_raw_df[self.target_column].values
        test_target = test_raw_df[self.target_column].values
        
        # ========== SCALE TARGET ==========
        self.scaler_manager.fit_target(train_target)
        train_target_scaled = self.scaler_manager.transform_target(train_target)
        val_target_scaled = self.scaler_manager.transform_target(val_target)
        test_target_scaled = self.scaler_manager.transform_target(test_target)
        
        # ========== PROCESS FEATURES ==========
        train_features_scaled = None
        val_features_scaled = None
        test_features_scaled = None
        
        if feature_cols and self.config.get('use_features', True):
            print("  Processing features...")
            
            train_features = train_raw_df[feature_cols].values
            val_features = val_raw_df[feature_cols].values
            test_features = test_raw_df[feature_cols].values
            
            n_features = len(feature_cols)
            train_features_scaled = np.zeros_like(train_features, dtype=np.float32)
            val_features_scaled = np.zeros_like(val_features, dtype=np.float32)
            test_features_scaled = np.zeros_like(test_features, dtype=np.float32)
            
            # Scale each feature independently (fit on train only)
            for i, col in enumerate(feature_cols):
                scaler = self.scaler_manager.fit_feature(col, train_features[:, i])
                train_features_scaled[:, i] = scaler.transform(train_features[:, i].reshape(-1, 1)).flatten()
                val_features_scaled[:, i] = self.scaler_manager.transform_feature(col, val_features[:, i])
                test_features_scaled[:, i] = self.scaler_manager.transform_feature(col, test_features[:, i])
        
        # ========== ADD FOURIER FEATURES ==========
        fourier_train = None
        fourier_val = None
        fourier_test = None
        
        if self.config.get('use_features', True):
            print("  Adding Fourier features...")
            fourier_train, self.fourier_params = self._add_fourier_features_correct(
                train_timestamps, fit=True
            )
            fourier_val, _ = self._add_fourier_features_correct(
                val_timestamps, fit=False, params=self.fourier_params
            )
            fourier_test, _ = self._add_fourier_features_correct(
                test_timestamps, fit=False, params=self.fourier_params
            )
        
        # ========== COMBINE TARGET AND FEATURES ==========
        combined_train = self._combine_target_and_features(
            train_target_scaled, train_features_scaled, fourier_train
        )
        combined_val = self._combine_target_and_features(
            val_target_scaled, val_features_scaled, fourier_val
        )
        combined_test = self._combine_target_and_features(
            test_target_scaled, test_features_scaled, fourier_test
        )
        
        print("  Creating sequences with proper temporal alignment...")
        splits = self._create_sequences_aligned(
            combined_train, combined_val, combined_test,
            train_target_scaled, val_target_scaled, test_target_scaled
        )
        
        splits['train_raw'] = train_target
        splits['val_raw'] = val_target
        splits['test_raw'] = test_target
        splits['train_timestamps'] = train_timestamps
        splits['val_timestamps'] = val_timestamps
        splits['test_timestamps'] = test_timestamps
        splits['feature_names'] = feature_cols
        splits['target_scaler'] = self.scaler_manager.target_scaler
        splits['scaler_manager'] = self.scaler_manager
        splits['target_column'] = self.target_column
        splits['country_code'] = country_code
        
        splits['best_epoch'] = None
        
        print(f"  Final shapes:")
        print(f"    X_train: {splits['X_train'].shape}")
        print(f"    X_val: {splits['X_val'].shape}")
        print(f"    X_test: {splits['X_test'].shape}")
        print(f"    Features per timestep: {splits['X_train'].shape[2]}")
        
        return splits
    
    def _handle_missing_values(self, df, fit=True):
        """
        Handle missing values with statistics from training data only.
        
        Args:
            df: DataFrame to process
            fit: If True, fit statistics; if False, use stored statistics
        
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].isna().any():
                if fit:
                    # Store statistics for future use
                    self.feature_stats[col] = {
                        'median': df_clean[col].median(),
                        'std': df_clean[col].std()
                    }
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    # Use stored statistics
                    if col in self.feature_stats:
                        df_clean[col] = df_clean[col].fillna(self.feature_stats[col]['median'])
                    else:
                        df_clean[col] = df_clean[col].fillna(0)
        
        return df_clean
    
    def _add_fourier_features_correct(self, timestamps, fit=True, params=None):
        """
        Add Fourier features for seasonality - CORRECT VERSION.
        
        Uses the proper angular frequency: 2π * k * t / period
        No scaling of time that would distort the period.
        
        Scientific justification:
        - Daily periodicity (24h) from human activity patterns
        - Weekly periodicity (168h) from work/weekend cycles
        
        Args:
            timestamps: DatetimeIndex
            fit: If True, fit parameters; if False, use provided params
            params: Previously fitted parameters (only used for t_start reference)
        
        Returns:
            fourier_features: array of Fourier features
            params: fitted parameters (t_start for reference)
        """
        # Convert timestamps to hours since start
        t_hours = (timestamps - timestamps[0]).total_seconds() / 3600.0
        
        if fit:
            params = {
                't_start': t_hours[0]
            }
        
        # FIXED: No scaling of time - use raw hours
        # Angular frequency: 2π * k * t / period
        features = []
        periods = self.config.get('fourier_periods', [24, 168])
        max_harmonic = self.config.get('n_fourier_terms', 4)
        
        for period in periods:
            for k in range(1, max_harmonic + 1):
                # Correct Fourier terms: sin(2π * k * t / period), cos(2π * k * t / period)
                angular = 2 * np.pi * k * t_hours / period
                features.append(np.sin(angular))
                features.append(np.cos(angular))
        
        return np.column_stack(features), params
    
    def _combine_target_and_features(self, target, features=None, fourier=None):
        """
        Combine target and features into a single array.
        
        IMPORTANT: Target is always the first column (index 0).
        This ensures the model always predicts the target, not features.
        
        Args:
            target: target values (1D array)
            features: optional feature array (2D)
            fourier: optional Fourier feature array (2D)
        
        Returns:
            combined: 2D array with target as first column
        """
        n_timesteps = len(target)
        
        if features is None and fourier is None:
            return target.reshape(-1, 1)
        
        # Start with target
        combined_list = [target.reshape(-1, 1)]
        
        # Add features if available
        if features is not None:
            combined_list.append(features)
        
        # Add Fourier features if available
        if fourier is not None:
            combined_list.append(fourier)
        
        return np.hstack(combined_list)
    
    def _create_sequences_aligned(self, train_data, val_data, test_data,
                                  train_target, val_target, test_target):
        """
        Create sequences with proper temporal alignment.
        
        CRITICAL: Validation and test sequences use context from previous splits
        to maintain temporal consistency.
        
        Args:
            train_data: training data (target + features)
            val_data: validation data
            test_data: test data
            train_target: training target (scaled)
            val_target: validation target (scaled)
            test_target: test target (scaled)
        
        Returns:
            dict with X_train, X_val, X_test, y_train, y_val, y_test
        """
        seq_len = self.config['sequence_length']
        horizon = self.config['forecast_horizon']
        
        # ========== TRAIN SEQUENCES ==========
        X_train, y_train = create_sequences_vectorized(train_data, seq_len, horizon)
        
        if X_train is None:
            raise ValueError(f"Not enough training data. Need at least {seq_len + horizon} samples.")
        
        # ========== VALIDATION SEQUENCES ==========
        # Add context from training data
        val_with_context = np.concatenate([train_data[-seq_len:], val_data])
        X_val, y_val = create_sequences_vectorized(val_with_context, seq_len, horizon)
        
        # ========== TEST SEQUENCES ==========
        # Add context from validation data
        test_with_context = np.concatenate([val_data[-seq_len:], test_data])
        X_test, y_test = create_sequences_vectorized(test_with_context, seq_len, horizon)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }


# ============================================================================
# SEQUENCE LENGTH TESTER
# ============================================================================

class SequenceLengthTester:
    """
    Test different sequence lengths to find optimal value.
    
    Scientific justification:
    - 24h: captures daily patterns
    - 48h: captures two-day patterns
    - 72h: captures three-day patterns
    - 168h: captures weekly patterns
    """
    
    def __init__(self, config, data_processor, country_code):
        self.config = config
        self.data_processor = data_processor
        self.country_code = country_code
        self.results = {}
    
    def test_all(self):
        """Test all sequence lengths in config."""
        seq_lengths = self.config.get('test_sequence_lengths', [24, 48, 72, 168])
        horizon = self.config['forecast_horizon']
        
        print(f"\n{'='*60}")
        print(f"TESTING SEQUENCE LENGTHS FOR {self.country_code}")
        print(f"{'='*60}")
        
        best_rmse = float('inf')
        best_seq_len = self.config['sequence_length']
        
        for seq_len in seq_lengths:
            print(f"\nTesting sequence_length = {seq_len}")
            
            # Update config
            test_config = self.config.copy()
            test_config['sequence_length'] = seq_len
            
            try:
                # Prepare splits with this sequence length
                processor = DataProcessor(test_config)
                processor.data = self.data_processor.data
                splits = processor.prepare_splits(self.country_code)
                
                # Train simple model for quick evaluation
                from tensorflow import keras
                model = keras.Sequential([
                    keras.layers.LSTM(16, input_shape=(seq_len, splits['X_train'].shape[2])),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(horizon)
                ])
                
                model.compile(optimizer='adam', loss='mse')
                
                # Train for few epochs
                history = model.fit(
                    splits['X_train'], splits['y_train'],
                    validation_data=(splits['X_val'], splits['y_val']),
                    epochs=20,
                    batch_size=32,
                    verbose=0,
                    shuffle=False
                )
                
                # Evaluate
                val_loss = min(history.history['val_loss'])
                
                self.results[seq_len] = {
                    'val_loss': float(val_loss),
                    'best_epoch': int(np.argmin(history.history['val_loss']) + 1)
                }
                
                if val_loss < best_rmse:
                    best_rmse = val_loss
                    best_seq_len = seq_len
                
                print(f"  Validation loss: {val_loss:.4f}")
                print(f"  Best epoch: {self.results[seq_len]['best_epoch']}")
                
            except Exception as e:
                print(f"  Error testing seq_len={seq_len}: {e}")
                self.results[seq_len] = {'val_loss': float('inf'), 'error': str(e)}
        
        print(f"\nBest sequence length: {best_seq_len} with validation loss {best_rmse:.4f}")
        
        return best_seq_len, self.results


# ============================================================================
# STRONG BASELINES WITH PROPER TUNING
# ============================================================================

class StrongBaselines:
    """
    Strong statistical baselines for forecasting comparison.
    
    Each baseline is tuned on validation data to ensure fair comparison
    with deep learning models.
    """
    
    def __init__(self, config):
        self.config = config
        self.best_params = {}
    
    def run_all(self, splits, horizon):
        """
        Run all baselines with proper tuning.
        
        Args:
            splits: data splits from DataProcessor
            horizon: forecast horizon
        
        Returns:
            dict: baseline results
            array: aligned test targets
        """
        print("\n" + "="*60)
        print("RUNNING STRONG BASELINES WITH TUNING")
        print("="*60)
        
        train_raw = splits['train_raw']
        val_raw = splits['val_raw']
        test_raw = splits['test_raw']
        
        n_predict = len(test_raw) - horizon + 1
        if n_predict <= 0:
            print(f"  Warning: Not enough test data. n_predict={n_predict}")
            return {}, None
        
        print(f"  Test samples: {len(test_raw)}, Predictions: {n_predict}")
        
        # Align test targets for multi-horizon evaluation
        y_test_aligned = np.zeros((n_predict, horizon))
        for h in range(horizon):
            start_idx = h
            end_idx = start_idx + n_predict
            if end_idx <= len(test_raw):
                y_test_aligned[:, h] = test_raw[start_idx:end_idx]
            else:
                y_test_aligned[:, h] = test_raw[start_idx:]
        
        results = {}
        
        # 0. Persistence baseline
        y_pred_persistence = self._persistence_baseline(test_raw, n_predict, horizon)
        results['persistence'] = {
            'predictions': y_pred_persistence,
            'rmse': np.sqrt(np.mean((y_pred_persistence - y_test_aligned) ** 2)),
            'tuned': False,
            'description': 'Naive persistence: y(t+h) = y(t)'
        }
        print(f"  Persistence - RMSE: {results['persistence']['rmse']:.2f}")
        
        # 1. Seasonal naive
        y_pred_sn = self._seasonal_naive(train_raw, n_predict, horizon)
        results['seasonal_naive'] = {
            'predictions': y_pred_sn,
            'rmse': np.sqrt(np.mean((y_pred_sn - y_test_aligned) ** 2)),
            'tuned': False,
            'description': 'Seasonal naive: last observed season'
        }
        print(f"  Seasonal naive - RMSE: {results['seasonal_naive']['rmse']:.2f}")
        
        # 2. Drift method
        y_pred_drift = self._drift_method(train_raw, n_predict, horizon)
        results['drift'] = {
            'predictions': y_pred_drift,
            'rmse': np.sqrt(np.mean((y_pred_drift - y_test_aligned) ** 2)),
            'tuned': False,
            'description': 'Drift method: linear extrapolation'
        }
        print(f"  Drift method - RMSE: {results['drift']['rmse']:.2f}")
        
        # 3. ARIMA with tuning
        try:
            y_pred_arima, best_order = self._tuned_arima(
                train_raw, val_raw, n_predict, horizon
            )
            results['arima_tuned'] = {
                'predictions': y_pred_arima,
                'rmse': np.sqrt(np.mean((y_pred_arima - y_test_aligned) ** 2)),
                'tuned': True,
                'best_params': {'order': best_order},
                'description': 'ARIMA with tuning on validation'
            }
            print(f"  ARIMA (tuned) - RMSE: {results['arima_tuned']['rmse']:.2f}, order={best_order}")
        except Exception as e:
            print(f"  ARIMA failed: {e}")
        
        # 4. ETS with tuning
        try:
            y_pred_ets, best_period = self._tuned_ets(
                train_raw, val_raw, n_predict, horizon
            )
            results['ets_tuned'] = {
                'predictions': y_pred_ets,
                'rmse': np.sqrt(np.mean((y_pred_ets - y_test_aligned) ** 2)),
                'tuned': True,
                'best_params': {'seasonal_period': best_period},
                'description': 'Exponential Smoothing with tuning'
            }
            print(f"  ETS (tuned) - RMSE: {results['ets_tuned']['rmse']:.2f}, period={best_period}")
        except Exception as e:
            print(f"  ETS failed: {e}")
        
        # 5. Prophet
        if PROPHET_AVAILABLE:
            try:
                y_pred_prophet = self._prophet_forecast(
                    train_raw, splits['train_timestamps'], n_predict, horizon
                )
                results['prophet'] = {
                    'predictions': y_pred_prophet,
                    'rmse': np.sqrt(np.mean((y_pred_prophet - y_test_aligned) ** 2)),
                    'tuned': False,
                    'description': 'Facebook Prophet'
                }
                print(f"  Prophet - RMSE: {results['prophet']['rmse']:.2f}")
            except Exception as e:
                print(f"  Prophet failed: {e}")
        
        # 6. XGBoost with no leakage
        try:
            y_pred_xgb = self._tuned_xgboost(
                splits, n_predict, horizon
            )
            results['xgboost_tuned'] = {
                'predictions': y_pred_xgb,
                'rmse': np.sqrt(np.mean((y_pred_xgb - y_test_aligned) ** 2)),
                'tuned': True,
                'description': 'XGBoost with lag features'
            }
            print(f"  XGBoost (tuned) - RMSE: {results['xgboost_tuned']['rmse']:.2f}")
        except Exception as e:
            print(f"  XGBoost error: {e}")
        
        return results, y_test_aligned
    
    def _persistence_baseline(self, test_raw, n_predict, horizon):
        """Persistence baseline: y(t+h) = y(t)"""
        predictions = np.zeros((n_predict, horizon))
        for i in range(n_predict):
            last_observed = test_raw[i] if i < len(test_raw) else test_raw[-1]
            predictions[i, :] = last_observed
        return predictions
    
    def _seasonal_naive(self, y_train, n_predict, horizon, season=24):
        """Seasonal naive forecast using last observed season."""
        if len(y_train) < season:
            return np.tile(y_train[-1], (n_predict, horizon))
        
        last_season = y_train[-season:]
        predictions = np.zeros((n_predict, horizon))
        for i in range(n_predict):
            for h in range(horizon):
                idx = (i + h) % season
                predictions[i, h] = last_season[idx]
        
        return predictions
    
    def _drift_method(self, y_train, n_predict, horizon):
        """Drift method (linear extrapolation)."""
        x_train = np.arange(len(y_train))
        slope = (y_train[-1] - y_train[0]) / max(1, len(y_train) - 1)
        intercept = y_train[0]
        
        predictions = np.zeros((n_predict, horizon))
        for i in range(n_predict):
            for h in range(horizon):
                predictions[i, h] = intercept + slope * (len(y_train) + i + h)
        
        return predictions
    
    def _tuned_arima(self, train_raw, val_raw, n_predict, horizon):
        """ARIMA with hyperparameter tuning on validation data."""
        best_rmse = float('inf')
        best_order = (1,1,1)
        best_forecast = None
        
        p_values = [0, 1, 2]
        d_values = [0, 1]
        q_values = [0, 1, 2]
        
        train_val = np.concatenate([train_raw, val_raw])
        
        n_val_predict = len(val_raw) - horizon + 1
        if n_val_predict <= 0:
            return np.tile(train_val[-1], (n_predict, horizon)), (1,1,1)
        
        for p, d, q in itertools.product(p_values, d_values, q_values):
            try:
                model = ARIMA(train_raw, order=(p, d, q))
                fitted = model.fit()
                forecast = fitted.forecast(steps=n_val_predict + horizon)
                
                pred_val = np.zeros((n_val_predict, horizon))
                for i in range(n_val_predict):
                    pred_val[i] = forecast[i:i+horizon]
                
                y_val_aligned = np.zeros((n_val_predict, horizon))
                for h in range(horizon):
                    y_val_aligned[:, h] = val_raw[h:h+n_val_predict]
                
                rmse = np.sqrt(np.mean((pred_val - y_val_aligned) ** 2))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = (p, d, q)
                    
                    model_full = ARIMA(train_val, order=best_order)
                    fitted_full = model_full.fit()
                    forecast_full = fitted_full.forecast(steps=n_predict + horizon)
                    
                    best_forecast = np.zeros((n_predict, horizon))
                    for i in range(n_predict):
                        best_forecast[i] = forecast_full[i:i+horizon]
            except:
                continue
        
        if best_forecast is None:
            best_forecast = np.tile(train_val[-1], (n_predict, horizon))
        
        return best_forecast, best_order
    
    def _tuned_ets(self, train_raw, val_raw, n_predict, horizon):
        """Exponential Smoothing with hyperparameter tuning."""
        best_rmse = float('inf')
        best_period = 24
        best_forecast = None
        
        periods = [24, 168]
        train_val = np.concatenate([train_raw, val_raw])
        n_val_predict = len(val_raw) - horizon + 1
        
        if n_val_predict <= 0:
            return np.tile(train_val[-1], (n_predict, horizon)), 24
        
        for period in periods:
            try:
                model = ExponentialSmoothing(
                    train_raw,
                    seasonal_periods=period,
                    trend='add',
                    seasonal='add'
                )
                fitted = model.fit()
                forecast = fitted.forecast(n_val_predict + horizon)
                
                pred_val = np.zeros((n_val_predict, horizon))
                for i in range(n_val_predict):
                    pred_val[i] = forecast[i:i+horizon]
                
                y_val_aligned = np.zeros((n_val_predict, horizon))
                for h in range(horizon):
                    y_val_aligned[:, h] = val_raw[h:h+n_val_predict]
                
                rmse = np.sqrt(np.mean((pred_val - y_val_aligned) ** 2))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_period = period
                    
                    model_full = ExponentialSmoothing(
                        train_val,
                        seasonal_periods=period,
                        trend='add',
                        seasonal='add'
                    )
                    fitted_full = model_full.fit()
                    forecast_full = fitted_full.forecast(n_predict + horizon)
                    
                    best_forecast = np.zeros((n_predict, horizon))
                    for i in range(n_predict):
                        best_forecast[i] = forecast_full[i:i+horizon]
            except:
                continue
        
        if best_forecast is None:
            best_forecast = np.tile(train_val[-1], (n_predict, horizon))
        
        return best_forecast, best_period
    
    def _prophet_forecast(self, train_raw, train_timestamps, n_predict, horizon):
        """Prophet forecast."""
        df_train = pd.DataFrame({
            'ds': train_timestamps,
            'y': train_raw
        })
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        model.fit(df_train)
        
        future = model.make_future_dataframe(periods=n_predict + horizon, freq='h')
        forecast = model.predict(future)
        y_pred_raw = forecast['yhat'].values[-n_predict - horizon:]
        
        y_pred = np.zeros((n_predict, horizon))
        for i in range(n_predict):
            y_pred[i] = y_pred_raw[i:i+horizon]
        
        return y_pred
    
    def _tuned_xgboost(self, splits, n_predict, horizon):
        """XGBoost with lag features and no leakage."""
        def create_lag_features(series, timestamps, lags=[24, 48]):
            df = pd.DataFrame({
                'target': series,
                'timestamp': timestamps
            })
            
            for lag in lags:
                if lag < len(df):
                    df[f'lag_{lag}'] = df['target'].shift(lag)
            
            df['hour'] = df['timestamp'].dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df = df.dropna()
            
            feature_cols = [c for c in df.columns if c not in ['target', 'timestamp', 'hour']]
            return df[feature_cols].values, df['target'].values, feature_cols
        
        X_train, y_train, feat_names = create_lag_features(
            splits['train_raw'], splits['train_timestamps']
        )
        X_val, y_val, _ = create_lag_features(
            splits['val_raw'], splits['val_timestamps']
        )
        X_test, y_test, _ = create_lag_features(
            splits['test_raw'], splits['test_timestamps']
        )
        
        if len(X_train) == 0 or len(X_val) == 0:
            return np.zeros((n_predict, horizon))
        
        best_rmse = float('inf')
        best_model = None
        
        for n_est in [100, 200]:
            for depth in [4, 6]:
                for lr in [0.03, 0.05]:
                    model = xgb.XGBRegressor(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        random_state=BASE_SEED,
                        verbosity=0,
                        n_jobs=1,
                        tree_method='hist'
                    )
                    model.fit(X_train, y_train)
                    y_pred_val = model.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model
        
        if best_model is None:
            return np.zeros((n_predict, horizon))
        
        y_pred = np.zeros((n_predict, horizon))
        if len(X_test) >= n_predict:
            y_pred[:, 0] = best_model.predict(X_test[:n_predict])
        else:
            y_pred[:, 0] = splits['test_raw'][-1]
        
        for h in range(1, horizon):
            y_pred[:, h] = y_pred[:, h-1]
        
        return y_pred


# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================

class BaseDLModel:
    """
    Base class for all deep learning models.
    
    SCIENTIFIC DESIGN:
    - All metrics computed on original scale after inverse transform
    - Early stopping prevents overfitting
    - Consistent evaluation across all model variants
    - Best epoch saved for reproducibility
    """
    
    def __init__(self, input_shape, config):
        self.input_shape = input_shape
        self.config = config
        self.model = None
        self.seed = config.get('seed', BASE_SEED)
        self.best_epoch = None
        
    def build(self):
        """Build model architecture - to be implemented by subclasses."""
        raise NotImplementedError
    
    def compile(self):
        if self.model is None:
            self.build()
        
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=self.config.get('gradient_clip', 1.0)
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train model with early stopping.
        
        FIXED: Lower patience (5) to prevent overfitting.
        FIXED: Save best epoch for reporting.
        """
        patience = self.config.get('patience', 5)
        
        class BestEpochCallback(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best_epoch = 0
                self.best_val_loss = float('inf')
            
            def on_epoch_end(self, epoch, logs=None):
                if logs and 'val_loss' in logs:
                    if logs['val_loss'] < self.best_val_loss:
                        self.best_val_loss = logs['val_loss']
                        self.best_epoch = epoch + 1
        
        best_epoch_callback = BestEpochCallback()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=self.config.get('min_delta', 0.001),
                restore_best_weights=True,
                mode='min'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                mode='min'
            ),
            best_epoch_callback
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        self.best_epoch = best_epoch_callback.best_epoch
        
        return history
    
    def predict(self, X):
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test, scaler):
        y_pred_scaled = self.predict(X_test)
        
        n_samples = y_test.shape[0]
        horizon = self.config['forecast_horizon']
        
        y_test_flat = y_test.reshape(-1, 1)
        y_pred_flat = y_pred_scaled.reshape(-1, 1)
        
        y_test_orig = scaler.inverse_transform(y_test_flat).flatten().reshape(n_samples, horizon)
        y_pred_orig = scaler.inverse_transform(y_pred_flat).flatten().reshape(n_samples, horizon)
        
        metrics = self._calculate_metrics(y_test_orig, y_pred_orig)
        metrics = self._add_confidence_intervals(y_test_orig, y_pred_orig, metrics)
        metrics['best_epoch'] = self.best_epoch
        
        return metrics, y_test_orig, y_pred_orig
    
    def _calculate_metrics(self, y_true, y_pred):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        
        denominator = (np.abs(y_true_flat) + np.abs(y_pred_flat)) / 2
        denominator = np.maximum(denominator, 0.01 * np.mean(np.abs(y_true_flat)))
        smape = np.mean(2 * np.abs(y_true_flat - y_pred_flat) / denominator) * 100
        
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        per_horizon_rmse = []
        for h in range(y_true.shape[1]):
            rmse_h = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
            per_horizon_rmse.append(float(rmse_h))
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'smape': float(smape),
            'r2': float(r2),
            'per_horizon_rmse': per_horizon_rmse
        }
    
    def _add_confidence_intervals(self, y_true, y_pred, metrics, n_bootstrap=500):
        errors = (y_true - y_pred).flatten()
        
        def rmse_from_errors(e):
            return np.sqrt(np.mean(e ** 2))
        
        block_size = self.config.get('seasonal_period', 24)
        n = len(errors)
        n_blocks = int(np.ceil(n / block_size))
        bootstrap_rmses = []
        
        for _ in range(n_bootstrap):
            blocks = []
            for _ in range(n_blocks):
                start = np.random.randint(0, max(1, n - block_size))
                blocks.append(errors[start:start+block_size])
            bootstrap_errors = np.concatenate(blocks)[:n]
            bootstrap_rmses.append(rmse_from_errors(bootstrap_errors))
        
        metrics['rmse_ci_lower'] = float(np.percentile(bootstrap_rmses, 2.5))
        metrics['rmse_ci_upper'] = float(np.percentile(bootstrap_rmses, 97.5))
        
        return metrics


class SimpleLSTM(BaseDLModel):
    """Simple LSTM with single layer - reduced size for your data."""
    
    def build(self):
        inputs = keras.Input(shape=self.input_shape)
        
        x = layers.LSTM(
            self.config['lstm_units_1'],  # 32
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
            recurrent_dropout=self.config['recurrent_dropout'],
            kernel_initializer='glorot_uniform'
        )(inputs)
        
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)  # 0.3
        
        x = layers.Dense(
            self.config['dense_units_1'],  # 16
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
            kernel_initializer='he_normal'
        )(x)
        x = layers.Dropout(self.config['dropout_rate'] / 2)(x)
        
        x = layers.Dense(
            self.config['dense_units_2'],  # 8
            activation='relu',
            kernel_initializer='he_normal'
        )(x)
        x = layers.Dropout(self.config['dropout_rate'] / 2)(x)
        
        outputs = layers.Dense(self.config['forecast_horizon'], kernel_initializer='glorot_uniform')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)


class EnhancedLSTM(BaseDLModel):
    """Enhanced LSTM with proper temporal attention."""
    
    def build(self):
        inputs = keras.Input(shape=self.input_shape)
        
        if self.config.get('use_bidirectional', False):
            x = layers.Bidirectional(
                layers.LSTM(
                    self.config['lstm_units_1'],
                    return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                    recurrent_dropout=self.config['recurrent_dropout'],
                    kernel_initializer='glorot_uniform'
                )
            )(inputs)
        else:
            x = layers.LSTM(
                self.config['lstm_units_1'],
                return_sequences=True,
                kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                recurrent_dropout=self.config['recurrent_dropout'],
                kernel_initializer='glorot_uniform'
            )(inputs)
        
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        x = layers.LSTM(
            self.config['lstm_units_2'],
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
            recurrent_dropout=self.config['recurrent_dropout'],
            kernel_initializer='glorot_uniform'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        if self.config.get('use_attention', False):
            attention_scores = layers.Dense(1, activation='tanh')(x)
            attention_scores = layers.Flatten()(attention_scores)
            attention_weights = layers.Activation('softmax', name='attention_weights')(attention_scores)
            attention_weights = layers.Reshape((-1, 1))(attention_weights)
            x = layers.multiply([x, attention_weights])
            x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name='attention_context')(x)
        else:
            x = layers.GlobalAveragePooling1D()(x)
        
        x = layers.Dense(
            self.config['dense_units_1'],
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
            kernel_initializer='he_normal'
        )(x)
        x = layers.Dropout(self.config['dropout_rate'] / 2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(
            self.config['dense_units_2'],
            activation='relu',
            kernel_initializer='he_normal'
        )(x)
        x = layers.Dropout(self.config['dropout_rate'] / 2)(x)
        
        outputs = layers.Dense(self.config['forecast_horizon'], kernel_initializer='glorot_uniform')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)


# ============================================================================
# MODEL FACTORY
# ============================================================================

class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(model_type, input_shape, config):
        if model_type == 'enhanced_lstm':
            return EnhancedLSTM(input_shape, config)
        elif model_type == 'simple_lstm':
            return SimpleLSTM(input_shape, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# STATISTICAL VALIDATOR
# ============================================================================

class StatisticalValidator:
    """Statistical tests for model comparison."""
    
    @staticmethod
    def compare_models(y_true, y_pred1, y_pred2, model1_name, model2_name, horizon=1):
        results = {}
        
        dm_stat, dm_p = diebold_mariano_test(y_true, y_pred1, y_pred2, h=horizon)
        results['dm_statistic'] = dm_stat
        results['dm_pvalue'] = dm_p
        results['dm_significant_005'] = dm_p < 0.05
        results['dm_significant_001'] = dm_p < 0.01
        
        e1 = np.abs(y_true - y_pred1)
        e2 = np.abs(y_true - y_pred2)
        w_stat, w_p = wilcoxon(e1 - e2)
        results['wilcoxon_statistic'] = w_stat
        results['wilcoxon_pvalue'] = w_p
        
        rmse1 = np.sqrt(np.mean((y_true - y_pred1) ** 2))
        rmse2 = np.sqrt(np.mean((y_true - y_pred2) ** 2))
        results['relative_improvement'] = (rmse1 - rmse2) / rmse1 * 100
        
        return results
    
    @staticmethod
    def multiple_comparison_correction(p_values, alpha=0.05, method='holm'):
        if method == 'bonferroni':
            n_tests = len(p_values)
            corrected_alpha = alpha / n_tests
            return [p < corrected_alpha for p in p_values]
        
        elif method == 'holm':
            return holm_bonferroni_correction(p_values, alpha)
        
        else:
            raise ValueError(f"Unknown correction method: {method}")


# ============================================================================
# SINGLE COUNTRY EXPERIMENT RUNNER
# ============================================================================

class SingleCountryRunner:
    """Runner for single country experiments."""
    
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.processor = DataProcessor(config)
        
    def run(self, country_code, test_sequence_lengths=False):
        print(f"\n{'='*70}")
        print(f"RUNNING EXPERIMENT FOR: {country_code}")
        print(f"{'='*70}")
        
        result_path = self.output_dir / f"final_results_{country_code}_{self.config['model_type']}.json"
        if self.config.get('skip_if_exists', True) and result_path.exists():
            print(f"  Results already exist for {country_code}, skipping...")
            with open(result_path, 'r') as f:
                return json.load(f)
        
        self.processor.load_data(self.config.get('data_path'))
        
        if not self.processor.check_country_availability(country_code):
            print(f"  {country_code}: Data not available or insufficient")
            return None
        
        try:
            if test_sequence_lengths:
                seq_tester = SequenceLengthTester(self.config, self.processor, country_code)
                best_seq_len, seq_results = seq_tester.test_all()
                self.config['best_sequence_length'] = best_seq_len
                self.config['sequence_length'] = best_seq_len
                
                seq_path = self.output_dir / f"sequence_length_test_{country_code}.json"
                with open(seq_path, 'w') as f:
                    json.dump(seq_results, f, indent=2)
            
            splits = self.processor.prepare_splits(country_code)
            
            baselines, y_test_aligned = StrongBaselines(self.config).run_all(
                splits, self.config['forecast_horizon']
            )
            
            baseline_path = self.output_dir / f"baselines_{country_code}.json"
            with open(baseline_path, 'w') as f:
                baseline_summary = {}
                for name, data in baselines.items():
                    if 'rmse' in data:
                        baseline_summary[name] = {
                            'rmse': float(data['rmse']),
                            'tuned': data.get('tuned', False),
                            'best_params': data.get('best_params', {}),
                            'description': data.get('description', '')
                        }
                json.dump(baseline_summary, f, indent=2)
            
            print(f"\n  Running {self.config['n_seeds']} seeds for {country_code}...")
            seed_results = []
            
            for seed_idx, seed in enumerate(self.config['seed_list']):
                print(f"\n    Seed {seed_idx + 1}/{self.config['n_seeds']} (seed={seed})")
                
                set_seed(seed, deterministic=self.config.get('deterministic', True))
                check_memory_and_clear(force=True)
                
                input_shape = splits['X_train'].shape[1:]
                model = ModelFactory.create_model(
                    self.config['model_type'], input_shape, self.config
                )
                model.build()
                model.compile()
                
                start_time = datetime.now()
                history = model.train(
                    splits['X_train'], splits['y_train'],
                    splits['X_val'], splits['y_val']
                )
                train_time = (datetime.now() - start_time).total_seconds()
                
                metrics, y_test_orig, y_pred_orig = model.evaluate(
                    splits['X_test'], splits['y_test'],
                    splits['target_scaler']
                )
                
                test_results = self._run_statistical_tests(
                    y_test_orig, y_pred_orig, baselines, splits
                )
                
                pred_df = pd.DataFrame({
                    'actual': y_test_orig.flatten(),
                    'predicted': y_pred_orig.flatten(),
                    'seed': seed
                })
                pred_path = self.output_dir / f"predictions_{country_code}_{self.config['model_type']}_seed{seed_idx}.csv"
                pred_df.to_csv(pred_path, index=False)
                
                seed_results.append({
                    'seed': seed,
                    'seed_index': seed_idx,
                    'metrics': metrics,
                    'test_results': test_results,
                    'train_time': train_time,
                    'best_epoch': metrics.get('best_epoch')
                })
                
                print(f"\n    Seed {seed_idx + 1} Results:")
                print(f"      RMSE: {metrics['rmse']:.2f} [{metrics['rmse_ci_lower']:.2f}, {metrics['rmse_ci_upper']:.2f}]")
                print(f"      MAE: {metrics['mae']:.2f}")
                print(f"      R²: {metrics['r2']:.4f}")
                print(f"      Best epoch: {metrics.get('best_epoch', 'N/A')}")
                
                del model
                tf.keras.backend.clear_session()
                for _ in range(3):
                    gc.collect()
                check_memory_and_clear(force=True)
            
            aggregated = self._aggregate_seed_results(seed_results)
            
            results = {
                'country': country_code,
                'model_type': self.config['model_type'],
                'n_seeds': self.config['n_seeds'],
                'seed_list': self.config['seed_list'],
                'aggregated': aggregated,
                'by_seed': seed_results,
                'baselines': {k: v['rmse'] for k, v in baselines.items() if 'rmse' in v},
                'baseline_details': {k: {'rmse': float(v['rmse']), 'description': v.get('description', '')} 
                                     for k, v in baselines.items() if 'rmse' in v},
                'best_baseline': min([v['rmse'] for k, v in baselines.items() if 'rmse' in v]),
                'best_baseline_name': min([(v['rmse'], k) for k, v in baselines.items() if 'rmse' in v])[1],
                'sequence_length_tested': test_sequence_lengths,
                'config': {k: str(v) for k, v in self.config.items() if not isinstance(v, (dict, list))}
            }
            
            best_baseline = results['best_baseline']
            results['improvement'] = (best_baseline - aggregated['rmse_mean']) / best_baseline * 100
            
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            try:
                self._save_visualizations(
                    country_code, history, y_test_orig, y_pred_orig, aggregated
                )
            except Exception as e:
                print(f"    Visualization error: {e}")
            
            print(f"\n  {country_code} completed: RMSE = {aggregated['rmse_mean']:.2f} ± {aggregated['rmse_std']:.2f}")
            
            return results
            
        except Exception as e:
            print(f"  Error for {country_code}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _run_statistical_tests(self, y_true, y_pred, baselines, splits):
        results = {}
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        p_values = []
        baseline_names = []
        
        horizon = self.config['forecast_horizon']
        
        for name, data in baselines.items():
            if 'predictions' in data:
                y_base_flat = data['predictions'].flatten()
                min_len = min(len(y_true_flat), len(y_base_flat))
                
                if min_len > 0:
                    dm_stat, dm_p = diebold_mariano_test(
                        y_true_flat[:min_len],
                        y_pred_flat[:min_len],
                        y_base_flat[:min_len],
                        h=horizon
                    )
                    
                    results[f'dm_vs_{name}_stat'] = dm_stat
                    results[f'dm_vs_{name}_p'] = dm_p
                    
                    p_values.append(dm_p)
                    baseline_names.append(name)
        
        if p_values:
            corrected = holm_bonferroni_correction(p_values, alpha=0.05)
            for i, name in enumerate(baseline_names):
                results[f'significant_vs_{name}'] = corrected[i]
        
        return results
    
    def _aggregate_seed_results(self, seed_results):
        aggregated = {}
        
        metrics_keys = ['rmse', 'mae', 'smape', 'r2']
        
        for key in metrics_keys:
            values = [r['metrics'][key] for r in seed_results]
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_min'] = float(np.min(values))
            aggregated[f'{key}_max'] = float(np.max(values))
        
        horizon = self.config['forecast_horizon']
        for h in range(horizon):
            values = [r['metrics']['per_horizon_rmse'][h] for r in seed_results]
            aggregated[f'rmse_h{h+1}_mean'] = float(np.mean(values))
            aggregated[f'rmse_h{h+1}_std'] = float(np.std(values))
        
        train_times = [r['train_time'] for r in seed_results]
        aggregated['train_time_mean'] = float(np.mean(train_times))
        aggregated['train_time_std'] = float(np.std(train_times))
        
        best_epochs = [r.get('best_epoch', 0) for r in seed_results if r.get('best_epoch')]
        if best_epochs:
            aggregated['best_epoch_mean'] = float(np.mean(best_epochs))
            aggregated['best_epoch_std'] = float(np.std(best_epochs))
        
        return aggregated
    
    def _save_visualizations(self, country, history, y_true, y_pred, metrics):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Forecasting Results - {country}', fontsize=16)
        
        if hasattr(history, 'history'):
            axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
            axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0, 0].axvline(x=metrics.get('best_epoch_mean', 0), color='green', linestyle='--', 
                               label=f"Best epoch ({metrics.get('best_epoch_mean', 'N/A'):.0f})")
            axes[0, 0].set_title('Learning Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss (MSE)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        n_plot = min(200, len(y_true))
        axes[0, 1].plot(y_true[:n_plot, 0], label='Actual', alpha=0.7, linewidth=1.5)
        axes[0, 1].plot(y_pred[:n_plot, 0], label='Predicted', alpha=0.7, linewidth=1.5)
        axes[0, 1].set_title('Predictions (h=1)')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Consumption')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        errors = (y_pred - y_true).flatten()
        axes[0, 2].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].axvline(x=0, color='red', linestyle='--')
        axes[0, 2].set_title('Error Distribution')
        axes[0, 2].set_xlabel('Error')
        axes[0, 2].set_ylabel('Frequency')
        
        horizons = range(1, self.config['forecast_horizon'] + 1)
        rmse_means = [metrics[f'rmse_h{h}_mean'] for h in horizons]
        rmse_stds = [metrics[f'rmse_h{h}_std'] for h in horizons]
        
        axes[1, 0].bar(horizons, rmse_means, yerr=rmse_stds, capsize=5)
        axes[1, 0].set_title('RMSE by Horizon (Mean ± Std over seeds)')
        axes[1, 0].set_xlabel('Horizon')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(y_true.flatten(), y_pred.flatten(), alpha=0.3, s=1)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 1].set_title('Actual vs Predicted')
        axes[1, 1].set_xlabel('Actual')
        axes[1, 1].set_ylabel('Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(errors[:n_plot*self.config['forecast_horizon']], alpha=0.5)
        axes[1, 2].axhline(y=0, color='red', linestyle='--')
        axes[1, 2].fill_between(range(len(errors[:n_plot*self.config['forecast_horizon']])),
                                 -metrics['rmse_std'], metrics['rmse_std'],
                                 alpha=0.2, color='gray')
        axes[1, 2].set_title('Residuals with Uncertainty')
        axes[1, 2].set_xlabel('Time Step')
        axes[1, 2].set_ylabel('Residual')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        vis_path = self.output_dir / f"visualizations_{country}_{self.config['model_type']}.png"
        plt.savefig(vis_path, dpi=150)
        plt.close()
        print(f"    Visualizations saved to {vis_path}")


# ============================================================================
# MULTI-COUNTRY EXPERIMENT RUNNER - FIXED FOR ALL COUNTRIES
# ============================================================================

class MultiCountryRunner:
    """Runner for multi-country experiments - processes ALL available countries."""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path("results")
        self.output_dir.mkdir(exist_ok=True)
        self.single_runner = SingleCountryRunner(config, self.output_dir)
        self.results = {}
        
    def find_available_countries(self):
        print("\n" + "="*70)
        print("FINDING AVAILABLE COUNTRIES")
        print("="*70)
        
        processor = DataProcessor(self.config)
        processor.load_data(self.config.get('data_path'))
        
        available = []
        countries = self.config.get('countries', ALL_COUNTRIES)
        
        # FIXED: Process ALL countries (max_countries = None means all)
        max_countries = self.config.get('max_countries')
        if max_countries is not None:
            countries = countries[:max_countries]
            print(f"  Limiting to {max_countries} countries as per config")
        
        for country in tqdm(countries, desc="Checking countries"):
            if processor.check_country_availability(country):
                available.append(country)
        
        print(f"\nFound {len(available)} available countries out of {len(countries)} checked")
        print(f"Available: {', '.join(available)}")
        
        return available
    
    def run_all(self, test_sequence_lengths=False):
        print("\n" + "="*70)
        print("MULTI-COUNTRY EXPERIMENT - PROCESSING ALL COUNTRIES")
        print("="*70)
        print("NOTE: This will process ALL available countries and may take several hours/days.")
        print("You can monitor progress in the results/ folder.\n")
        
        available_countries = self.find_available_countries()
        
        if not available_countries:
            print("No countries with available data found.")
            return {}
        
        print(f"\nProcessing {len(available_countries)} countries...")
        
        successful = []
        failed = []
        
        for i, country in enumerate(available_countries):
            print(f"\n{'='*70}")
            print(f"[{i+1}/{len(available_countries)}] Processing {country}...")
            print(f"{'='*70}")
            
            result = self.single_runner.run(country, test_sequence_lengths=test_sequence_lengths)
            
            if result:
                self.results[country] = result
                successful.append(country)
                print(f"{country} completed successfully")
            else:
                failed.append(country)
                print(f"{country} failed")
            
            # Force garbage collection between countries
            gc.collect()
            tf.keras.backend.clear_session()
            
            # Save intermediate results
            self._save_intermediate_results(successful, failed)
        
        self._generate_summary(successful, failed)
        
        return self.results
    
    def _save_intermediate_results(self, successful, failed):
        """Save intermediate results in case of crash."""
        intermediate = {
            'timestamp': datetime.now().isoformat(),
            'successful': successful,
            'failed': failed,
            'results': {
                country: {
                    'rmse_mean': self.results[country]['aggregated']['rmse_mean'],
                    'r2_mean': self.results[country]['aggregated']['r2_mean'],
                    'improvement': self.results[country].get('improvement', 0)
                }
                for country in successful if country in self.results
            }
        }
        
        path = self.output_dir / "intermediate_results.json"
        with open(path, 'w') as f:
            json.dump(intermediate, f, indent=2)
    
    def _generate_summary(self, successful, failed):
        print("\n" + "="*70)
        print("MULTI-COUNTRY SUMMARY - ALL COUNTRIES")
        print("="*70)
        print(f"Successful: {len(successful)} countries")
        print(f"Failed: {len(failed)} countries")
        
        if successful:
            country_results = []
            for country in successful:
                if country in self.results:
                    res = self.results[country]
                    country_results.append({
                        'country': country,
                        'rmse': res['aggregated']['rmse_mean'],
                        'rmse_std': res['aggregated']['rmse_std'],
                        'mae': res['aggregated']['mae_mean'],
                        'r2': res['aggregated']['r2_mean'],
                        'improvement': res.get('improvement', 0),
                        'best_baseline': res.get('best_baseline_name', 'unknown'),
                        'best_epoch': res['aggregated'].get('best_epoch_mean', 0)
                    })
            
            # Sort by R² (best first)
            country_results.sort(key=lambda x: x['r2'], reverse=True)
            
            print(f"\n{'Country':<6} {'RMSE':<10} {'+/-':<3} {'MAE':<10} {'R²':<8} {'Impr':<8} {'BestEpoch':<8}")
            print("-" * 80)
            
            for cr in country_results:
                impr_symbol = "+" if cr['improvement'] > 0 else "-"
                print(f"{cr['country']:<6} {cr['rmse']:<10.1f} +/-{cr['rmse_std']:<4.1f} "
                      f"{cr['mae']:<10.1f} {cr['r2']:<8.4f} {impr_symbol}{cr['improvement']:<6.1f}% "
                      f"{cr['best_epoch']:<8.0f}")
            
            # Calculate aggregates
            rmse_values = [cr['rmse'] for cr in country_results]
            r2_values = [cr['r2'] for cr in country_results]
            impr_values = [cr['improvement'] for cr in country_results]
            
            print(f"\nAggregates over {len(country_results)} countries (mean +/- std):")
            print(f"  RMSE: {np.mean(rmse_values):.2f} +/- {np.std(rmse_values):.2f}")
            print(f"  R²: {np.mean(r2_values):.4f} +/- {np.std(r2_values):.4f}")
            print(f"  Improvement over best baseline: {np.mean(impr_values):.1f}% +/- {np.std(impr_values):.1f}%")
            
            # Save final summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'config': {k: str(v) for k, v in self.config.items()},
                'successful': successful,
                'failed': failed,
                'total_countries': len(successful) + len(failed),
                'results': {
                    country: {
                        'rmse_mean': self.results[country]['aggregated']['rmse_mean'],
                        'rmse_std': self.results[country]['aggregated']['rmse_std'],
                        'r2_mean': self.results[country]['aggregated']['r2_mean'],
                        'improvement': self.results[country].get('improvement', 0),
                        'best_baseline': self.results[country].get('best_baseline_name', 'unknown'),
                        'best_epoch': self.results[country]['aggregated'].get('best_epoch_mean', 0)
                    }
                    for country in successful
                }
            }
            
            summary_path = self.output_dir / "multi_country_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nSummary saved to: {summary_path}")
            print(f"All results are in: {self.output_dir}")
        
        if failed:
            print(f"\nFailed countries: {', '.join(failed)}")


# ============================================================================
# CONFIGURATION FILE GENERATOR
# ============================================================================

def create_default_config(config_path='config.yaml'):
    """Create a default configuration file."""
    config = Config.get_default_config()
    
    config_yaml = yaml.dump(config, default_flow_style=False)
    
    with open(config_path, 'w') as f:
        f.write("# Energy Forecasting System Configuration\n")
        f.write("# Scientific justifications for key parameters:\n")
        f.write("# - sequence_length: 24h captures daily consumption patterns\n")
        f.write("# - forecast_horizon: 6h for operational planning\n")
        f.write("# - fourier_periods: [24,168] daily and weekly seasonality\n")
        f.write("# - patience: 5 to prevent overfitting\n")
        f.write("# - dropout_rate: 0.3 for strong regularization\n")
        f.write("# - lstm_units: [32,16] small architecture to avoid overfitting\n")
        f.write("# - n_seeds: 5 for robust multi-seed evaluation\n")
        f.write("# - seed_list: explicit list of random seeds\n")
        f.write("# - max_features_per_country: limit features to prevent overfitting\n")
        f.write("# - max_countries: null = process ALL available countries\n\n")
        f.write(config_yaml)
    
    print(f"Default configuration saved to {config_path}")
    return config_path


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with configuration file support."""
    if len(sys.argv) == 0:
        sys.argv = ['']
    
    parser = argparse.ArgumentParser(description='Energy Forecasting System')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--create-config', action='store_true',
                        help='Create default configuration file and exit')
    parser.add_argument('--country', type=str, default=None,
                        help='Country code (overrides config, for single country mode)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model type (overrides config)')
    parser.add_argument('--multi', action='store_true',
                        help='Run multi-country analysis (ALL countries)')
    parser.add_argument('--test-seq-lengths', action='store_true',
                        help='Test multiple sequence lengths to find optimal')
    parser.add_argument('--max-countries', type=int, default=None,
                        help='Maximum number of countries to process (None = all)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data file (overrides config)')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config(args.config)
        return
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
    else:
        config = Config.get_default_config()
        print(f"Config file {args.config} not found. Using default configuration.")
        print("Consider running with --create-config to generate a default config file.")
    
    if args.data_path:
        config['data_path'] = args.data_path
    elif config.get('data_path') is None:
        # Default to your data path - FIXED: Using forward slashes
        config['data_path'] = "C:/Users/Zahara/Documents/Zoom/europe_energy_forecast/data/europe_energy_real.csv"
        print(f"Using default data path: {config['data_path']}")
    
    # Verify data path exists
    if not os.path.exists(config['data_path']):
        print(f"WARNING: Data file not found at: {config['data_path']}")
        print("Will create sample dataset instead.")
    
    if args.country:
        config['country_code'] = args.country
        config['run_multi_country'] = False
    if args.model:
        config['model_type'] = args.model
    if args.multi:
        config['run_multi_country'] = True
    if args.max_countries is not None:
        config['max_countries'] = args.max_countries
    
    if 'seed_list' not in config:
        config['seed_list'] = [BASE_SEED + i for i in range(config.get('n_seeds', N_SEEDS))]
    
    print("\n" + "="*70)
    print("ENERGY FORECASTING SYSTEM v16.1")
    print("="*70)
    print("\nConfiguration:")
    for key, value in config.items():
        if not isinstance(value, dict) and not isinstance(value, list):
            print(f"  {key}: {value}")
    print(f"  seed_list: {config['seed_list']}")
    
    # Check if running multi-country
    if config.get('run_multi_country', False):
        max_msg = "ALL countries" if config.get('max_countries') is None else f"max {config['max_countries']} countries"
        print(f"\nRunning multi-country mode - will process {max_msg}")
        print("This may take several hours/days depending on number of countries")
        print("Results will be saved to: results/")
        
        runner = MultiCountryRunner(config)
        results = runner.run_all(test_sequence_lengths=args.test_seq_lengths)
    else:
        # Single country mode
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nRunning single country mode for: {config['country_code']}")
        print(f"Results will be saved to: {output_dir}")
        
        runner = SingleCountryRunner(config, output_dir)
        results = runner.run(config['country_code'], test_sequence_lengths=args.test_seq_lengths)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
