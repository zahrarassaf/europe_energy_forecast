import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

class AdvancedStatisticalAnalyzer:
    def __init__(self, data):
        self.data = data
    
    def comprehensive_stationarity_test(self, column):
        """ADF and KPSS tests for stationarity"""
        # ADF Test
        adf_result = adfuller(self.data[column].dropna())
        
        # KPSS Test  
        kpss_result = kpss(self.data[column].dropna())
        
        return {
            'adf': {'statistic': adf_result[0], 'p_value': adf_result[1]},
            'kpss': {'statistic': kpss_result[0], 'p_value': kpss_result[1]},
            'is_stationary': adf_result[1] < 0.05 and kpss_result[1] > 0.05
        }
    
    def seasonal_decomposition_analysis(self, column, period=365):
        """Decompose time series into components"""
        decomposition = seasonal_decompose(
            self.data[column].dropna(), 
            period=period, 
            model='additive'
        )
        return decomposition
    
    def cross_country_causality(self, target_country, cause_country, max_lag=7):
        """Granger causality between countries"""
        test_data = self.data[[target_country, cause_country]].dropna()
        result = grangercausalitytests(test_data, max_lag, verbose=False)
        
        p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
        best_lag = np.argmin(p_values) + 1
        best_p_value = min(p_values)
        
        return {
            'causes': best_p_value < 0.05,
            'best_lag': best_lag,
            'p_value': best_p_value
        }
