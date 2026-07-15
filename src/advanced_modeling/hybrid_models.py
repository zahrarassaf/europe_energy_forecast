"""
hybrid_ensemble_v2.py - Improved version with no data leakage, better structure,
and comprehensive error handling.

This module provides a hybrid ensemble forecasting system for European energy load data.
It combines multiple ML models (XGBoost, LightGBM, RandomForest, GradientBoosting, Ridge)
using weighted ensemble methods with proper time series cross-validation.
"""

import logging
import warnings
import re
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# For caching
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class ModelConfig:
    """
    Configuration for all models in the ensemble with validation.
    
    Attributes:
        xgb_params: XGBoost hyperparameters
        lgb_params: LightGBM hyperparameters
        rf_params: RandomForest hyperparameters
        gb_params: GradientBoosting hyperparameters
        ridge_alpha: Ridge regularization parameter
        use_meta_learner: Whether to use meta-learner for ensemble weights
        meta_learner_alpha: Meta-learner regularization
        min_weight: Minimum weight for ensemble members
    """
    
    # XGBoost Configuration
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1
    })
    
    # LightGBM Configuration
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    })
    
    # Random Forest Configuration
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # Gradient Boosting Configuration
    gb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    })
    
    # Ridge Configuration
    ridge_alpha: float = 1.0
    
    # Ensemble Configuration
    use_meta_learner: bool = True
    meta_learner_alpha: float = 1.0
    min_weight: float = 0.01
    
    # Time features
    use_time_features: bool = True
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 24, 48, 168])
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if self.meta_learner_alpha <= 0:
            raise ValueError("meta_learner_alpha must be positive")
        if not 0 <= self.min_weight <= 1:
            raise ValueError("min_weight must be between 0 and 1")
        if not self.lags:
            raise ValueError("lags cannot be empty")


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Data configuration
    test_size: float = 0.2
    validation_size: float = 0.1
    min_samples: int = 1000
    n_samples_per_country: int = 30000
    
    # Cross-validation
    n_cv_splits: int = 5
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Caching
    use_cache: bool = True
    cache_dir: str = "models/cache"
    
    # Reports
    output_dir: str = "outputs"
    save_plots: bool = True
    save_predictions: bool = True
    
    def __post_init__(self):
        """Create necessary directories."""
        for directory in [self.cache_dir, self.output_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Handler
# ============================================================================

class DataHandler:
    """
    Handles data loading and preprocessing without data leakage.
    
    This class ensures that all feature engineering is done with proper
    time series awareness to prevent look-ahead bias.
    """
    
    def __init__(
        self, 
        data_path: str, 
        config: ModelConfig,
        training_config: TrainingConfig
    ):
        """
        Initialize the DataHandler.
        
        Args:
            data_path: Path to the CSV data file
            config: Model configuration
            training_config: Training configuration
        """
        self.data_path = Path(data_path)
        self.config = config
        self.training_config = training_config
        self._data_cache: Optional[pd.DataFrame] = None
        self.logger = logging.getLogger(f"{__name__}.DataHandler")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load data with caching.
        
        Args:
            force_reload: If True, force reload from disk
            
        Returns:
            Loaded DataFrame
        """
        if force_reload or self._data_cache is None:
            self.logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            
            # Clean column names
            df.columns = [col.strip().replace(' ', '_') for col in df.columns]
            
            # Store cache
            if self.training_config.use_cache:
                self._data_cache = df
            
            return df
        
        return self._data_cache
    
    def detect_countries(self, n_samples: int = 10000) -> List[str]:
        """
        Detect all available countries in the dataset.
        
        Args:
            n_samples: Number of samples to scan for detection
            
        Returns:
            List of country codes
        """
        self.logger.info("Detecting available countries...")
        
        df = self.load_data()
        if len(df) > n_samples:
            df = df.iloc[:n_samples]
        
        pattern = r'([A-Z]{2})_load_actual_entsoe_transparency'
        countries = []
        
        for col in df.columns:
            match = re.match(pattern, col)
            if match:
                country_code = match.group(1)
                if country_code not in countries:
                    countries.append(country_code)
        
        self.logger.info(f"Found {len(countries)} countries: {', '.join(sorted(countries))}")
        return sorted(countries)
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        country: str, 
        split_idx: int
    ) -> Tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray]]:
        """
        Prepare features without data leakage.
        
        This is the critical method that ensures no look-ahead bias.
        Training data only uses information available at each time point.
        
        Args:
            df: Input DataFrame
            country: Country code
            split_idx: Index where training data ends
            
        Returns:
            Tuple of (X features, y target, lag_1_values for baseline)
        """
        target_col = f"{country}_load_actual_entsoe_transparency"
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Work with a copy to avoid modifying original
        df = df.copy()
        df['load'] = df[target_col].copy()
        
        # Split into train and test to prevent leakage
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Initialize feature dictionaries
        train_features = {}
        test_features = {}
        
        # --- Lag features (without leakage) ---
        for lag in self.config.lags:
            lag_col = f'lag_{lag}'
            
            # For training: normal shift within training data
            train_df[lag_col] = train_df['load'].shift(lag)
            
            # For testing: use last training value for first test sample
            test_df[lag_col] = test_df['load'].shift(lag)
            
            # If lag is longer than test data available, use training values
            if lag > 1:
                # Fill first few test values with rolling average from train
                last_train_values = train_df['load'].iloc[-min(lag, len(train_df)):].values
                
                for i in range(min(lag, len(test_df))):
                    if pd.isna(test_df[lag_col].iloc[i]):
                        if i < len(last_train_values):
                            test_df[lag_col].iloc[i] = last_train_values[i]
            
            # Forward fill any remaining NaNs
            train_df[lag_col] = train_df[lag_col].ffill()
            test_df[lag_col] = test_df[lag_col].ffill()
            
            # Store
            train_features[lag_col] = train_df[lag_col].values
            test_features[lag_col] = test_df[lag_col].values
        
        # --- Lag 1 for baseline ---
        train_df['lag_1'] = train_df['load'].shift(1).ffill()
        test_df['lag_1'] = test_df['load'].shift(1)
        test_df['lag_1'] = test_df['lag_1'].ffill()
        
        # If first test value is NaN, use last training value
        if pd.isna(test_df['lag_1'].iloc[0]):
            test_df['lag_1'].iloc[0] = train_df['load'].iloc[-1]
        
        lag_1_train = train_df['lag_1'].values
        lag_1_test = test_df['lag_1'].values
        
        # --- Time features (without leakage) ---
        if self.config.use_time_features and 'utc_timestamp' in df.columns:
            try:
                for split_name, split_df in [('train', train_df), ('test', test_df)]:
                    # Parse timestamps
                    split_df['timestamp'] = pd.to_datetime(split_df['utc_timestamp'])
                    
                    # Hour features
                    split_df['hour'] = split_df['timestamp'].dt.hour
                    split_df['hour_sin'] = np.sin(2 * np.pi * split_df['hour'] / 24)
                    split_df['hour_cos'] = np.cos(2 * np.pi * split_df['hour'] / 24)
                    
                    # Day of week features
                    split_df['day_of_week'] = split_df['timestamp'].dt.dayofweek
                    split_df['day_sin'] = np.sin(2 * np.pi * split_df['day_of_week'] / 7)
                    split_df['day_cos'] = np.cos(2 * np.pi * split_df['day_of_week'] / 7)
                    
                    # Month features
                    split_df['month'] = split_df['timestamp'].dt.month
                    split_df['month_sin'] = np.sin(2 * np.pi * split_df['month'] / 12)
                    split_df['month_cos'] = np.cos(2 * np.pi * split_df['month'] / 12)
                    
                    # Store
                    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
                    for feat in time_features:
                        if split_name == 'train':
                            train_features[feat] = split_df[feat].values
                        else:
                            test_features[feat] = split_df[feat].values
                            
            except Exception as e:
                self.logger.warning(f"Error creating time features: {e}")
        
        # --- Combine features ---
        feature_names = sorted(list(train_features.keys()))
        
        if not feature_names:
            raise ValueError("No features could be created")
        
        X_train = np.column_stack([train_features[name] for name in feature_names])
        X_test = np.column_stack([test_features[name] for name in feature_names])
        
        y_train = train_df['load'].values
        y_test = test_df['load'].values
        
        # Combine lag_1 for baseline
        lag_1_values = np.concatenate([lag_1_train, lag_1_test])
        
        # --- Clean data (remove NaNs and infinities) ---
        # For training
        train_valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train) & ~np.isinf(y_train)
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        
        # For testing
        test_valid = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test) & ~np.isinf(y_test)
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]
        lag_1_test = lag_1_values[split_idx:][test_valid]
        
        # Combine clean data
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        lag_1_values = np.concatenate([lag_1_train[train_valid], lag_1_test])
        
        return X, y, lag_1_values


# ============================================================================
# Model Trainer
# ============================================================================

class ModelTrainer:
    """
    Trains multiple models with proper cross-validation and caching.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        cache_manager: Optional['CacheManager'] = None
    ):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            cache_manager: Optional cache manager for model caching
        """
        self.model_config = model_config
        self.training_config = training_config
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(f"{__name__}.ModelTrainer")
        self.scaler = StandardScaler()
    
    def train_all_models(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_test: npt.NDArray,
        country: str
    ) -> Tuple[Dict[str, Any], Dict[str, npt.NDArray], Dict[str, npt.NDArray]]:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            country: Country code for caching
            
        Returns:
            Tuple of (model_objects, training_predictions, test_predictions)
        """
        # Check cache first
        model_objects = {}
        train_predictions = {}
        test_predictions = {}
        
        # Scale features for Ridge
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'XGBoost': xgb.XGBRegressor(**self.model_config.xgb_params),
            'LightGBM': lgb.LGBMRegressor(**self.model_config.lgb_params),
            'RandomForest': RandomForestRegressor(**self.model_config.rf_params),
            'GradientBoosting': GradientBoostingRegressor(**self.model_config.gb_params),
            'Ridge': Ridge(alpha=self.model_config.ridge_alpha, random_state=42)
        }
        
        # Train each model
        for name, model in models.items():
            try:
                # Check cache
                if self.cache_manager and self.training_config.use_cache:
                    cached_model = self.cache_manager.load_model(country, name)
                    if cached_model is not None:
                        self.logger.info(f"Loaded cached {name} for {country}")
                        model = cached_model
                    else:
                        # Train model
                        model = self._train_model(model, name, X_train, y_train, X_train_scaled)
                        # Save to cache
                        self.cache_manager.save_model(model, country, name)
                else:
                    # Train without cache
                    model = self._train_model(model, name, X_train, y_train, X_train_scaled)
                
                # Get predictions
                if name == 'Ridge':
                    pred_train = model.predict(X_train_scaled)
                    pred_test = model.predict(X_test_scaled)
                else:
                    pred_train = model.predict(X_train)
                    pred_test = model.predict(X_test)
                
                model_objects[name] = model
                train_predictions[name] = pred_train
                test_predictions[name] = pred_test
                
                self.logger.debug(f"Successfully trained {name} for {country}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {name} for {country}: {str(e)[:100]}")
                continue
        
        if not model_objects:
            raise RuntimeError(f"No models could be trained for {country}")
        
        return model_objects, train_predictions, test_predictions
    
    def _train_model(
        self,
        model: Any,
        name: str,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_train_scaled: npt.NDArray
    ) -> Any:
        """
        Train a single model with optional cross-validation.
        
        Args:
            model: Model instance
            name: Model name
            X_train: Training features
            y_train: Training targets
            X_train_scaled: Scaled features for Ridge
            
        Returns:
            Trained model
        """
        if self.training_config.n_cv_splits > 1:
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.training_config.n_cv_splits)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_train_cv = X_train[train_idx]
                y_train_cv = y_train[train_idx]
                X_val_cv = X_train[val_idx]
                y_val_cv = y_train[val_idx]
                
                # Train on fold
                if name == 'Ridge':
                    X_train_scaled_cv = self.scaler.fit_transform(X_train_cv)
                    X_val_scaled_cv = self.scaler.transform(X_val_cv)
                    model.fit(X_train_scaled_cv, y_train_cv)
                    pred = model.predict(X_val_scaled_cv)
                else:
                    model.fit(X_train_cv, y_train_cv)
                    pred = model.predict(X_val_cv)
                
                rmse = np.sqrt(mean_squared_error(y_val_cv, pred))
                cv_scores.append(rmse)
            
            self.logger.debug(f"{name} CV RMSE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
        
        # Final training on all data
        if name == 'Ridge':
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        
        return model


# ============================================================================
# Ensemble Builder
# ============================================================================

class EnsembleBuilder:
    """
    Builds weighted ensembles from multiple models.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the EnsembleBuilder.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EnsembleBuilder")
    
    def build_ensemble(
        self,
        train_predictions: Dict[str, npt.NDArray],
        test_predictions: Dict[str, npt.NDArray],
        y_train: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray, Dict[str, float]]:
        """
        Build weighted ensemble from model predictions.
        
        Args:
            train_predictions: Dictionary of training predictions from each model
            test_predictions: Dictionary of test predictions from each model
            y_train: Training targets
            
        Returns:
            Tuple of (ensemble_train_predictions, ensemble_test_predictions, weights)
        """
        model_names = list(train_predictions.keys())
        n_models = len(model_names)
        
        if n_models == 0:
            raise ValueError("No model predictions available")
        
        if n_models == 1:
            name = model_names[0]
            return (
                train_predictions[name],
                test_predictions[name],
                {name: 1.0}
            )
        
        # Calculate weights
        weights = self._calculate_weights(
            train_predictions,
            test_predictions,
            y_train,
            model_names
        )
        
        # Apply weights to get ensemble predictions
        ensemble_train = np.zeros_like(y_train)
        ensemble_test = np.zeros_like(list(test_predictions.values())[0])
        
        for i, name in enumerate(model_names):
            ensemble_train += weights[i] * train_predictions[name]
            ensemble_test += weights[i] * test_predictions[name]
        
        weight_dict = {name: float(weights[i]) for i, name in enumerate(model_names)}
        
        self.logger.debug(f"Ensemble weights: {weight_dict}")
        
        return ensemble_train, ensemble_test, weight_dict
    
    def _calculate_weights(
        self,
        train_predictions: Dict[str, npt.NDArray],
        test_predictions: Dict[str, npt.NDArray],
        y_train: npt.NDArray,
        model_names: List[str]
    ) -> npt.NDArray:
        """
        Calculate optimal weights for ensemble.
        
        Uses either meta-learner (Ridge) or correlation-based weighting.
        """
        n_models = len(model_names)
        predictions_matrix = np.column_stack([
            train_predictions[name] for name in model_names
        ])
        
        weights = None
        
        # Method 1: Meta-learner
        if self.config.use_meta_learner:
            try:
                meta_learner = Ridge(
                    alpha=self.config.meta_learner_alpha,
                    random_state=42
                )
                meta_learner.fit(predictions_matrix, y_train)
                weights = meta_learner.coef_
                
                # Ensure non-negative weights
                weights = np.maximum(weights, 0)
                
                # Apply minimum weight
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    weights = None
                    
            except Exception as e:
                self.logger.warning(f"Meta-learner failed: {e}")
                weights = None
        
        # Method 2: Correlation-based
        if weights is None:
            correlation_weights = []
            for name in model_names:
                corr = np.corrcoef(y_train, train_predictions[name])[0, 1]
                correlation_weights.append(max(0, corr))
            
            weights = np.array(correlation_weights)
            
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n_models) / n_models
        
        # Ensure minimum weight
        weights = np.maximum(weights, self.config.min_weight)
        weights = weights / np.sum(weights)
        
        return weights


# ============================================================================
# Cache Manager
# ============================================================================

class CacheManager:
    """
    Handles caching of trained models to disk.
    """
    
    def __init__(self, cache_dir: str):
        """
        Initialize the CacheManager.
        
        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.CacheManager")
    
    def get_cache_path(self, country: str, model_name: str) -> Path:
        """Get the cache path for a specific model."""
        return self.cache_dir / f"{country}_{model_name}.joblib"
    
    def load_model(self, country: str, model_name: str) -> Optional[Any]:
        """
        Load a model from cache.
        
        Args:
            country: Country code
            model_name: Name of the model
            
        Returns:
            Loaded model or None if not found
        """
        cache_path = self.get_cache_path(country, model_name)
        
        if cache_path.exists():
            try:
                model = joblib.load(cache_path)
                self.logger.info(f"Loaded {model_name} for {country} from cache")
                return model
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {country} {model_name}: {e}")
                return None
        
        return None
    
    def save_model(self, model: Any, country: str, model_name: str) -> bool:
        """
        Save a model to cache.
        
        Args:
            model: Model to save
            country: Country code
            model_name: Name of the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_path = self.get_cache_path(country, model_name)
            joblib.dump(model, cache_path)
            self.logger.debug(f"Saved {model_name} for {country} to cache")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save cache for {country} {model_name}: {e}")
            return False
    
    def clear_cache(self, country: Optional[str] = None):
        """
        Clear cache files.
        
        Args:
            country: If provided, only clear for specific country
        """
        pattern = f"{country}_" if country else ""
        for cache_file in self.cache_dir.glob(f"{pattern}*.joblib"):
            cache_file.unlink()
            self.logger.debug(f"Removed cache: {cache_file}")
    
    def get_cache_size(self) -> int:
        """Get total size of cache in bytes."""
        total = 0
        for cache_file in self.cache_dir.glob("*.joblib"):
            total += cache_file.stat().st_size
        return total


# ============================================================================
# Results Manager
# ============================================================================

class ResultsManager:
    """
    Manages and stores results from country analyses.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the ResultsManager.
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.ResultsManager")
    
    def store_result(self, country: str, result: Dict[str, Any]):
        """Store results for a country."""
        self.results[country] = result
    
    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored results."""
        return self.results
    
    def get_country_result(self, country: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific country."""
        return self.results.get(country)
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate a comparative report across countries.
        
        Returns:
            DataFrame with comparative results
        """
        if not self.results:
            self.logger.warning("No results to report")
            return pd.DataFrame()
        
        data = []
        for country, result in self.results.items():
            data.append({
                'Country': country,
                'Samples': result.get('total_samples', 0),
                'Train_Samples': result.get('train_samples', 0),
                'Test_Samples': result.get('test_samples', 0),
                'Load_Mean_MW': int(result.get('load_mean', 0)),
                'Load_Std_MW': int(result.get('load_std', 0)),
                'Ensemble_R2': result.get('ensemble_r2', 0),
                'Ensemble_RMSE_MW': int(result.get('ensemble_rmse', 0)),
                'Ensemble_MAE_MW': int(result.get('ensemble_mae', 0)),
                'Best_Model': result.get('best_individual_model', 'N/A'),
                'Baseline_R2': result.get('baseline_r2', 0),
                'Improvement_%': result.get('improvement_pct', 0)
            })
        
        df_report = pd.DataFrame(data)
        df_report = df_report.sort_values('Ensemble_R2', ascending=False)
        
        return df_report
    
    def save_report(self, filename: str = 'ensemble_report.csv'):
        """Save report to CSV."""
        report_df = self.generate_report()
        if not report_df.empty:
            filepath = self.output_dir / filename
            report_df.to_csv(filepath, index=False)
            self.logger.info(f"Report saved to {filepath}")
    
    def save_detailed_results(self, filename: str = 'detailed_results.csv'):
        """
        Save detailed results for all countries.
        
        Args:
            filename: Output filename
        """
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        detailed_data = []
        for country, result in self.results.items():
            row = {
                'country': country,
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_samples': result.get('total_samples', 0),
                'train_samples': result.get('train_samples', 0),
                'test_samples': result.get('test_samples', 0),
                'load_mean_mw': result.get('load_mean', 0),
                'load_std_mw': result.get('load_std', 0),
                'load_min_mw': result.get('load_min', 0),
                'load_max_mw': result.get('load_max', 0),
                'best_individual_model': result.get('best_individual_model', 'N/A'),
                'ensemble_r2': result.get('ensemble_r2', 0),
                'ensemble_rmse_mw': result.get('ensemble_rmse', 0),
                'ensemble_mae_mw': result.get('ensemble_mae', 0),
                'ensemble_weights': str(result.get('ensemble_weights', {})),
                'baseline_r2': result.get('baseline_r2', 0),
                'baseline_rmse_mw': result.get('baseline_rmse', 0),
                'improvement_pct': result.get('improvement_pct', 0)
            }
            
            # Add individual model metrics
            for model_name, metrics in result.get('all_models', {}).items():
                row[f'{model_name}_r2'] = metrics.get('r2', 0)
                row[f'{model_name}_rmse'] = metrics.get('rmse', 0)
                row[f'{model_name}_mae'] = metrics.get('mae', 0)
            
            detailed_data.append(row)
        
        df_detailed = pd.DataFrame(detailed_data)
        filepath = self.output_dir / filename
        df_detailed.to_csv(filepath, index=False, encoding='utf-8')
        
        self.logger.info(f"Detailed results saved to {filepath}")
        return df_detailed


# ============================================================================
# Main HybridEnsembleForecaster Class
# ============================================================================

class HybridEnsembleForecaster:
    """
    Main class for hybrid ensemble forecasting.
    
    This class orchestrates the entire forecasting pipeline:
    1. Data loading and country detection
    2. Feature engineering without leakage
    3. Model training with cross-validation
    4. Ensemble building with weighted averaging
    5. Results collection and reporting
    """
    
    def __init__(
        self,
        data_path: str = 'data/europe_energy_real.csv',
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize the forecaster.
        
        Args:
            data_path: Path to data CSV file
            model_config: Model configuration (creates default if None)
            training_config: Training configuration (creates default if None)
        """
        # Configurations
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        
        # Initialize components
        self.data_handler = DataHandler(
            data_path,
            self.model_config,
            self.training_config
        )
        
        self.cache_manager = CacheManager(self.training_config.cache_dir)
        
        self.model_trainer = ModelTrainer(
            self.model_config,
            self.training_config,
            self.cache_manager
        )
        
        self.ensemble_builder = EnsembleBuilder(self.model_config)
        self.results_manager = ResultsManager(self.training_config.output_dir)
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.HybridEnsembleForecaster")
        
        self.logger.info("="*80)
        self.logger.info("MULTI-COUNTRY ENERGY LOAD FORECASTING - HYBRID ENSEMBLE V2")
        self.logger.info("="*80)
    
    def analyze_country(
        self,
        country: str,
        n_samples: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a single country with hybrid ensemble.
        
        Args:
            country: Country code (e.g., 'DE', 'FR')
            n_samples: Number of samples to use (default from config)
            
        Returns:
            Dictionary with analysis results or None if failed
        """
        if n_samples is None:
            n_samples = self.training_config.n_samples_per_country
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"ANALYZING: {country} (HYBRID ENSEMBLE V2)")
        self.logger.info(f"{'='*60}")
        
        try:
            # Load data
            df = self.data_handler.load_data()
            if len(df) > n_samples:
                df = df.iloc[:n_samples]
            
            # Check target column
            target_col = f"{country}_load_actual_entsoe_transparency"
            if target_col not in df.columns:
                self.logger.warning(f"Target column {target_col} not found for {country}")
                return None
            
            # Prepare features without leakage
            split_idx = int(len(df) * (1 - self.training_config.test_size))
            
            try:
                X, y, lag_1_values = self.data_handler.prepare_features(
                    df, country, split_idx
                )
            except Exception as e:
                self.logger.error(f"Feature preparation failed for {country}: {e}")
                return None
            
            # Check minimum samples
            if len(y) < self.training_config.min_samples:
                self.logger.warning(
                    f"Skipping {country}: insufficient data "
                    f"({len(y)} < {self.training_config.min_samples})"
                )
                return None
            
            # Split data
            train_size = int(len(X) * (1 - self.training_config.test_size))
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_test = X[train_size:]
            y_test = y[train_size:]
            
            # Get lag_1 for baseline
            lag_1_test = lag_1_values[train_size:]
            
            self.logger.info(f"Training samples: {len(y_train):,}")
            self.logger.info(f"Test samples: {len(y_test):,}")
            
            # Train all models
            self.logger.info("Training ensemble models...")
            model_objects, train_preds, test_preds = self.model_trainer.train_all_models(
                X_train, y_train, X_test, country
            )
            
            if not test_preds:
                self.logger.warning(f"No models trained for {country}")
                return None
            
            # Build ensemble
            ensemble_train, ensemble_test, weights = self.ensemble_builder.build_ensemble(
                train_preds, test_preds, y_train
            )
            
            # Calculate metrics
            metrics = {}
            
            # Individual model metrics
            for name, pred in test_preds.items():
                metrics[name] = {
                    'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                    'mae': mean_absolute_error(y_test, pred),
                    'r2': r2_score(y_test, pred)
                }
            
            # Ensemble metrics
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test))
            ensemble_mae = mean_absolute_error(y_test, ensemble_test)
            ensemble_r2 = r2_score(y_test, ensemble_test)
            
            metrics['Hybrid_Ensemble'] = {
                'rmse': ensemble_rmse,
                'mae': ensemble_mae,
                'r2': ensemble_r2
            }
            
            # Baseline (lag-1) metrics
            baseline_metrics = None
            if len(lag_1_test) == len(y_test):
                baseline_rmse = np.sqrt(mean_squared_error(y_test, lag_1_test))
                baseline_mae = mean_absolute_error(y_test, lag_1_test)
                baseline_r2 = r2_score(y_test, lag_1_test)
                baseline_metrics = {
                    'rmse': baseline_rmse,
                    'mae': baseline_mae,
                    'r2': baseline_r2
                }
            
            # Find best individual model
            best_model_name = min(
                [(name, m['rmse']) for name, m in metrics.items() if name != 'Hybrid_Ensemble'],
                key=lambda x: x[1]
            )[0]
            
            # Prepare result
            result = {
                'country': country,
                'total_samples': len(y),
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'load_mean': y.mean(),
                'load_std': y.std(),
                'load_min': y.min(),
                'load_max': y.max(),
                'best_individual_model': best_model_name,
                'ensemble_r2': ensemble_r2,
                'ensemble_rmse': ensemble_rmse,
                'ensemble_mae': ensemble_mae,
                'ensemble_weights': weights,
                'baseline_r2': baseline_metrics['r2'] if baseline_metrics else None,
                'baseline_rmse': baseline_metrics['rmse'] if baseline_metrics else None,
                'all_models': metrics,
                # Store predictions for future use
                '_predictions': {
                    'y_test': y_test,
                    'ensemble_test': ensemble_test,
                    'test_predictions': test_preds
                }
            }
            
            if baseline_metrics:
                improvement = ((baseline_metrics['rmse'] - ensemble_rmse) / 
                              baseline_metrics['rmse']) * 100
                result['improvement_pct'] = improvement
            
            # Store results
            self.results_manager.store_result(country, result)
            
            # Print summary
            self.logger.info(f"\nResults for {country} (HYBRID ENSEMBLE V2):")
            self.logger.info(f"  Samples: {len(y):,}")
            self.logger.info(f"  Ensemble R²: {ensemble_r2:.4f}")
            self.logger.info(f"  Ensemble RMSE: {ensemble_rmse:.1f} MW")
            self.logger.info(f"  Best Individual: {best_model_name}")
            self.logger.info(f"  Ensemble Weights: {weights}")
            if baseline_metrics:
                self.logger.info(f"  Baseline R²: {baseline_metrics['r2']:.4f}")
                self.logger.info(f"  Improvement: {improvement:.1f}%")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {country}: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def run_all_countries(
        self,
        max_countries: Optional[int] = None,
        n_samples: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run analysis for all detected countries.
        
        Args:
            max_countries: Maximum number of countries to analyze
            n_samples: Number of samples per country
            
        Returns:
            Dictionary with results for all countries
        """
        # Detect countries
        countries = self.data_handler.detect_countries()
        
        if max_countries:
            countries = countries[:max_countries]
            self.logger.info(f"Analyzing first {max_countries} countries")
        
        self.logger.info(f"\nAnalyzing {len(countries)} countries...")
        self.logger.info("="*60)
        
        # Process each country with progress bar
        for country in tqdm(countries, desc="Countries (Hybrid Ensemble V2)"):
            result = self.analyze_country(country, n_samples)
            if result:
                self.logger.info(f"✓ Completed analysis for {country}")
            else:
                self.logger.warning(f"✗ Failed to analyze {country}")
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"COMPLETED: Analyzed {len(self.results_manager.results)} countries successfully")
        self.logger.info("="*80)
        
        return self.results_manager.results
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comparative report across countries."""
        return self.results_manager.generate_report()
    
    def save_results(self, filename: str = 'ensemble_results.csv'):
        """Save results to CSV."""
        self.results_manager.save_detailed_results(filename)
    
    def save_report(self, filename: str = 'ensemble_report.csv'):
        """Save report to CSV."""
        self.results_manager.save_report(filename)
    
    def plot_results(self, country: Optional[str] = None, save: bool = True):
        """
        Plot prediction results for countries.
        
        Args:
            country: Specific country to plot, or None for all
            save: Whether to save plots to disk
        """
        results = self.results_manager.get_all_results()
        
        if not results:
            self.logger.warning("No results to plot")
            return
        
        if country:
            if country not in results:
                self.logger.warning(f"No results for country {country}")
                return
            countries_to_plot = [country]
        else:
            countries_to_plot = list(results.keys())
        
        for country in countries_to_plot:
            result = results[country]
            
            if '_predictions' not in result:
                self.logger.warning(f"No predictions stored for {country}")
                continue
            
            preds = result['_predictions']
            y_test = preds['y_test']
            ensemble_test = preds['ensemble_test']
            
            # Create plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Time series prediction
            ax = axes[0, 0]
            n_plot = min(500, len(y_test))
            ax.plot(y_test[:n_plot], label='Actual', linewidth=2)
            ax.plot(ensemble_test[:n_plot], label='Ensemble Prediction', linewidth=2, alpha=0.7)
            ax.set_title(f'{country} - Actual vs Predicted (First {n_plot} samples)')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Load (MW)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Scatter plot
            ax = axes[0, 1]
            ax.scatter(y_test, ensemble_test, alpha=0.3, s=1)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
            ax.set_title(f'{country} - Actual vs Predicted (Scatter)')
            ax.set_xlabel('Actual Load (MW)')
            ax.set_ylabel('Predicted Load (MW)')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Residuals
            ax = axes[1, 0]
            residuals = y_test - ensemble_test
            ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_title(f'{country} - Residual Distribution')
            ax.set_xlabel('Residual (MW)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Model comparison
            ax = axes[1, 1]
            models = [m for m in result['all_models'].keys() if m != 'Hybrid_Ensemble']
            rmse_values = [result['all_models'][m]['rmse'] for m in models]
            models.append('Hybrid_Ensemble')
            rmse_values.append(result['ensemble_rmse'])
            
            colors = ['skyblue'] * len(models)
            colors[-1] = 'gold'
            
            bars = ax.bar(models, rmse_values, color=colors, alpha=0.7)
            ax.set_title(f'{country} - RMSE Comparison')
            ax.set_xlabel('Model')
            ax.set_ylabel('RMSE (MW)')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, rmse_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}',
                       ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save:
                save_path = Path(self.training_config.output_dir) / f'{country}_ensemble_plot.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            plt.close()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main function for hybrid ensemble analysis.
    
    Command line usage:
        python hybrid_ensemble_v2.py --max-countries 5 --samples 30000 --output results.csv
    """
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Multi-Country Energy Load Forecasting (HYBRID ENSEMBLE V2)'
    )
    parser.add_argument(
        '--max-countries',
        type=int,
        default=None,
        help='Maximum number of countries to analyze'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=30000,
        help='Number of samples per country'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ensemble_results.csv',
        help='Output CSV file'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable model caching'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cache before running'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("EUROPEAN ENERGY LOAD FORECASTING - HYBRID ENSEMBLE V2")
    print("="*80)
    print("Improved version with:")
    print("  ✓ No data leakage")
    print("  ✓ Time series cross-validation")
    print("  ✓ Model caching")
    print("  ✓ Comprehensive error handling")
    print("="*80 + "\n")
    
    # Create configuration
    model_config = ModelConfig()
    training_config = TrainingConfig(
        n_samples_per_country=args.samples,
        use_cache=not args.no_cache
    )
    
    # Clear cache if requested
    if args.clear_cache:
        cache_manager = CacheManager(training_config.cache_dir)
        cache_manager.clear_cache()
        print("Cache cleared")
    
    # Initialize forecaster
    forecaster = HybridEnsembleForecaster(
        data_path='data/europe_energy_real.csv',
        model_config=model_config,
        training_config=training_config
    )
    
    # Run analysis
    results = forecaster.run_all_countries(
        max_countries=args.max_countries,
        n_samples=args.samples
    )
    
    if results:
        # Generate report
        report_df = forecaster.generate_report()
        
        if not report_df.empty:
            print("\n" + "="*80)
            print("COMPARATIVE ANALYSIS REPORT (HYBRID ENSEMBLE V2)")
            print("="*80)
            print(report_df.to_string(index=False))
        
        # Save results
        forecaster.save_results(args.output)
        forecaster.save_report('report_' + args.output)
        
        # Plot results for best country
        if not report_df.empty:
            best_country = report_df.iloc[0]['Country']
            forecaster.plot_results(best_country, save=True)
        
        print("\n" + "="*80)
        print(f"HYBRID ENSEMBLE V2 ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Successfully analyzed {len(results)} countries")
        print(f"Results saved to: {args.output}")
        print("="*80)




if __name__ == "__main__":
    main()
