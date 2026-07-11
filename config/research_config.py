"""
Configuration module for energy load forecasting project.
Handles all configuration parameters with validation and type checking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import os
from datetime import datetime


class CountryCode(Enum):
    """Standardized country codes for European energy market."""
    GERMANY = 'DE'
    FRANCE = 'FR' 
    ITALY = 'IT'
    SPAIN = 'ES'
    UK = 'UK'
    NETHERLANDS = 'NL'
    BELGIUM = 'BE'
    POLAND = 'PL'
    
    @classmethod
    def get_all_codes(cls) -> List[str]:
        return [country.value for country in cls]


class FeatureCategory(Enum):
    """Categorization of feature types for better organization."""
    LOAD = 'load'
    GENERATION = 'generation'
    PRICE = 'price'
    WEATHER = 'weather'
    TIME = 'time'


@dataclass(frozen=True)  # Immutable configuration
class ModelConfig:
    """Configuration for energy forecasting model."""
    
    # Target configuration
    target_country: CountryCode = CountryCode.GERMANY
    target_column: str = 'DE_load_actual_entsoe_transparency'
    
    # Data configuration
    countries: Tuple[str, ...] = field(default_factory=lambda: tuple(CountryCode.get_all_codes()))
    sequence_length: int = 24  # hours
    forecast_horizon: int = 24  # hours
    test_size: float = 0.2
    
    # Feature configuration with categories
    feature_columns: Tuple[str, ...] = field(default_factory=lambda: (
        # Target country features
        'DE_load_forecast_entsoe_transparency',
        'DE_solar_generation_actual',
        'DE_wind_generation_actual', 
        'DE_price_day_ahead',
        
        # Neighboring countries load
        'FR_load_actual_entsoe_transparency',
        'IT_load_actual_entsoe_transparency',
        'ES_load_actual_entsoe_transparency',
        'NL_load_actual_entsoe_transparency',
    ))
    
    # Feature engineering
    time_features: List[str] = field(default_factory=lambda: [
        'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_holiday'
    ])
    
    # Data validation
    min_sequence_length: int = 6
    max_sequence_length: int = 168  # 1 week
    min_test_size: float = 0.1
    max_test_size: float = 0.4
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not self._validate_sequence_length():
            raise ValueError(f"Sequence length must be between {self.min_sequence_length} and {self.max_sequence_length}")
        
        if not self._validate_test_size():
            raise ValueError(f"Test size must be between {self.min_test_size} and {self.max_test_size}")
        
        if not self._validate_forecast_horizon():
            raise ValueError("Forecast horizon cannot exceed sequence length")
    
    def _validate_sequence_length(self) -> bool:
        return self.min_sequence_length <= self.sequence_length <= self.max_sequence_length
    
    def _validate_test_size(self) -> bool:
        return self.min_test_size <= self.test_size <= self.max_test_size
    
    def _validate_forecast_horizon(self) -> bool:
        return self.forecast_horizon <= self.sequence_length


@dataclass(frozen=True)
class DataConfig:
    """Data source and preprocessing configuration."""
    
    data_source: str = "entsoe_transparency"
    data_version: str = "v1.0"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Preprocessing flags
    handle_missing: str = "interpolate"  # options: drop, interpolate, ffill
    normalize: bool = True
    scaling_method: str = "standard"  # options: standard, minmax, robust
    
    # Cache configuration
    use_cache: bool = True
    cache_dir: str = "./data/cache"
    
    def __post_init__(self):
        """Create cache directory if it doesn't exist."""
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)


@dataclass(frozen=True)
class ModelTrainingConfig:
    """Training configuration for the neural network."""
    
    # Architecture
    lstm_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.2
    use_attention: bool = True
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "huber"  # options: mse, mae, huber
    early_stopping_patience: int = 10
    validation_split: float = 0.1
    
    # Checkpoints
    save_best_model: bool = True
    model_checkpoint_dir: str = "./models/checkpoints"
    tensorboard_log_dir: str = "./logs"
    
    def __post_init__(self):
        """Create necessary directories."""
        for directory in [self.model_checkpoint_dir, self.tensorboard_log_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)


# Singleton configuration instances
config = ModelConfig()
data_config = DataConfig()
training_config = ModelTrainingConfig()
