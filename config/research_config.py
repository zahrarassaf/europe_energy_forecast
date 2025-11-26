from dataclasses import dataclass, field
from typing import List

@dataclass
class ResearchConfig:
    # Data parameters
    DATA_PATH: str = "data/europe_energy.csv"
    COUNTRIES: List[str] = field(default_factory=lambda: ['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE', 'PL'])
    TARGET_COUNTRY: str = 'DE'
    
    # Time parametersafrom dataclasses import dataclass

@dataclass
class ResearchConfig:
    DATA_PATH: str = "data/europe_energy.csv"
    COUNTRIES: list = None
    TARGET_COUNTRY: str = 'DE'
    TEST_SIZE: float = 0.2
    SEQUENCE_LENGTH: int = 30
    LAGS: list = None
    ROLLING_WINDOWS: list = None

    def __post_init__(self):
        if self.COUNTRIES is None:
            self.COUNTRIES = ['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE', 'PL']
        if self.LAGS is None:
            self.LAGS = [1, 7, 30]
        if self.ROLLING_WINDOWS is None:
            self.ROLLING_WINDOWS = [7, 30]

config = ResearchConfig()
    START_DATE: str = "2006-01-01"
    END_DATE: str = "2017-12-31"
    TEST_SIZE: float = 0.2
    FORECAST_HORIZON: int = 30
    
    # Model parameters
    SEQUENCE_LENGTH: int = 30
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    
    # Advanced features
    LAGS: List[int] = field(default_factory=lambda: [1, 7, 30])
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [7, 30])

config = ResearchConfig()
