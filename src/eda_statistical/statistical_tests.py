import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
from arch import arch_test
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedStatisticalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.results = {}
        
    def comprehensive_stationarity_tests(self, column='energy_consumption_mwh'):
        series = self.data[column]
        
        adf_result = adfuller(series, autolag='AIC')
        kpss_result = kpss(series, regression='c', nlags='auto')
        
        from arch.unitroot import PhillipsPerron
        pp_result = PhillipsPerron(series)
        
        self.results['stationarity'] = {
            'adf': {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'stationary': adf_result[1] < 0.05
            },
            'kpss': {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'stationary': kpss_result[1] > 0.05
            },
            'phillips_perron': {
                'statistic': pp_result.stat,
                'p_value': pp_result.pvalue,
                'stationary': pp_result.pvalue < 0.05
            }
        }
        
        return self.results['stationarity']
    
    def normality_test_suite(self, column='energy_consumption_mwh'):
        series = self.data[column]
        
        tests = {
            'shapiro_wilk': stats.shapiro(series),
            'jarque_bera': stats.jarque_bera(series),
            'anderson_darling': stats.anderson(series, dist='norm'),
            'dagostino_pearson': stats.normaltest(series)
        }
        
        self.results['normality'] = tests
        return tests
    
    def time_series_decomposition(self, column='energy_consumption_mwh', period=365):
        series = self.data[column]
        
        decomposition = seasonal_decompose(series, model='additive', period=period)
        
        seasonal_strength = max(0, 1 - (decomposition.resid.var() / 
                                       (decomposition.trend + decomposition.resid).var()))
        
        self.results['decomposition'] = {
            'decomposition': decomposition,
            'seasonal_strength': seasonal_strength,
            'trend_strength': max(0, 1 - (decomposition.resid.var() / 
                                        (decomposition.seasonal + decomposition.resid).var()))
        }
        
        return self.results['decomposition']
    
    def advanced_correlation_analysis(self):
        pearson_corr = self.data.corr(method='pearson')
        spearman_corr = self.data.corr(method='spearman')
        
        from statsmodels.tsa.stattools import ccf
        cross_correlations = {}
        target_col = 'energy_consumption_mwh'
        
        for col in self.data.columns:
            if col != target_col:
                cross_correlations[col] = ccf(self.data[target_col], self.data[col])[:30]
        
        self.results['correlation'] = {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'cross_correlation': cross_correlations
        }
        
        return self.results['correlation']
    
    def causality_analysis(self, target_col='energy_consumption_mwh', maxlag=4):
        causality_results = {}
        
        for col in self.data.columns:
            if col != target_col:
                try:
                    test_data = self.data[[target_col, col]].dropna()
                    granger_test = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                    
                    best_lag = min(granger_test.keys(), 
                                 key=lambda x: granger_test[x][0]['ssr_ftest'][1])
                    
                    causality_results[col] = {
                        'best_lag': best_lag,
                        'p_value': granger_test[best_lag][0]['ssr_ftest'][1],
                        'causes': granger_test[best_lag][0]['ssr_ftest'][1] < 0.05
                    }
                except:
                    causality_results[col] = {'error': 'Test failed'}
        
        self.results['causality'] = causality_results
        return causality_results
    
    def generate_statistical_report(self):
        print("=" * 80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        print("=" * 80)
        
        self.comprehensive_stationarity_tests()
        self.normality_test_suite()
        self.time_series_decomposition()
        self.advanced_correlation_analysis()
        self.causality_analysis()
        
        self._print_stationarity_results()
        self._print_normality_results()
        self._print_causality_results()
        
        return self.results
    
    def _print_stationarity_results(self):
        print("\nSTATIONARITY ANALYSIS:")
        print("-" * 40)
        stationarity = self.results['stationarity']
        
        for test_name, result in stationarity.items():
            if test_name != 'phillips_perron':
                print(f"{test_name.upper():<20}: p-value = {result['p_value']:.6f}, "
                      f"Stationary = {result['stationary']}")
    
    def _print_normality_results(self):
        print("\nNORMALITY TESTS:")
        print("-" * 40)
        normality = self.results['normality']
        
        for test_name, result in normality.items():
            if test_name == 'anderson_darling':
                print(f"{test_name:<20}: statistic = {result[0]:.4f}, "
                      f"Critical Values = {result[1]}")
            else:
                print(f"{test_name:<20}: statistic = {result[0]:.4f}, "
                      f"p-value = {result[1]:.6f}")
    
    def _print_causality_results(self):
        print("\nGRANGER CAUSALITY RESULTS:")
        print("-" * 40)
        causality = self.results['causality']
        
        for variable, result in causality.items():
            if 'p_value' in result:
                print(f"{variable:<25}: p-value = {result['p_value']:.6f}, "
                      f"Causes = {result['causes']} (lag {result['best_lag']})")

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
    sample_data.set_index('date', inplace=True)
    
    analyzer = AdvancedStatisticalAnalyzer(sample_data)
    report = analyzer.generate_statistical_report()
