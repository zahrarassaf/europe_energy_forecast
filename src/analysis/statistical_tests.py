import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

class AdvancedStatisticalTests:
    def __init__(self, data):
        self.data = data
    
    def stationarity_analysis(self, column):
        """Comprehensive stationarity tests"""
        result = adfuller(self.data[column].dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    def granger_causality(self, target_country, cause_country, max_lag=7):
        """Test if one country's data causes another"""
        test_data = self.data[[target_country, cause_country]].dropna()
        result = grangercausalitytests(test_data, max_lag, verbose=False)
        
        p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
        return min(p_values) < 0.05, p_values
    
    def seasonal_decomposition(self, column, period=365):
        """Decompose time series into trend, seasonal, residual"""
        decomposition = seasonal_decompose(self.data[column].dropna(), 
                                         period=period, model='additive')
        return decomposition
