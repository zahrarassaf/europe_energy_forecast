import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import jarque_bera, anderson, normaltest
import os
import sys
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.results = {}
    
    def _hampel_filter(self, series, window=10, n_sigmas=3):
        median = series.rolling(window=window, center=True).median()
        mad = series.rolling(window=window, center=True).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > n_sigmas
    
    def comprehensive_analysis(self, column_name):
        print(f"\n{'='*70}")
        print(f"STATISTICAL ANALYSIS: {column_name}")
        print(f"{'='*70}")
        
        if column_name not in self.data.columns:
            print(f"Error: {column_name} not found")
            return None
        
        series = self.data[column_name].dropna()
        if len(series) < 10:
            print(f"Warning: Only {len(series)} observations - skipping")
            return None
        
        print(f"\nObservations: {len(series):,}")
        if isinstance(series.index, pd.DatetimeIndex):
            print(f"Date range: {series.index.min()} to {series.index.max()}")
        
        analysis_results = {
            'column': column_name,
            'basic_stats': {},
            'stationarity_tests': {},
            'distribution_tests': {},
            'autocorrelation': {},
            'seasonality': {},
            'heteroskedasticity': {},
            'outliers': {},
            'summary': {}
        }
        
        print("\nBASIC STATISTICS:")
        print("-" * 40)
        
        basic_stats = {
            'count': len(series),
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'median': series.median(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75),
            'iqr': series.quantile(0.75) - series.quantile(0.25),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'cv': (series.std() / series.mean()) * 100 if series.mean() != 0 else np.nan
        }
        
        for stat_name, stat_value in basic_stats.items():
            if isinstance(stat_value, float):
                print(f"{stat_name:15s}: {stat_value:.4f}")
            else:
                print(f"{stat_name:15s}: {stat_value}")
        
        analysis_results['basic_stats'] = basic_stats
        
        print("\nSTATIONARITY TESTS:")
        print("-" * 40)
        
        try:
            adf_result = adfuller(series, autolag='AIC')
            adf_test = {
                'test_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            }
            print(f"ADF: stat={adf_test['test_statistic']:.4f}, p={adf_test['p_value']:.6f}, stationary={'YES' if adf_test['is_stationary'] else 'NO'}")
        except:
            adf_test = {'is_stationary': False}
        
        try:
            kpss_result = kpss(series, regression='c', nlags='auto')
            kpss_test = {
                'test_statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'is_stationary': kpss_result[1] > 0.05
            }
            print(f"KPSS: stat={kpss_test['test_statistic']:.4f}, p={kpss_test['p_value']:.6f}, stationary={'YES' if kpss_test['is_stationary'] else 'NO'}")
        except:
            kpss_test = {'is_stationary': False}
        
        if adf_test.get('is_stationary', False) and kpss_test.get('is_stationary', False):
            series_type = "I(0) - Stationary"
        elif adf_test.get('is_stationary', False) and not kpss_test.get('is_stationary', False):
            series_type = "Trend-Stationary"
        elif not adf_test.get('is_stationary', False) and not kpss_test.get('is_stationary', False):
            series_type = "I(1) - Needs Differencing"
        else:
            series_type = "Uncertain"
        
        print(f"\nSeries Type: {series_type}")
        analysis_results['stationarity_tests'] = {'adf': adf_test, 'kpss': kpss_test, 'series_type': series_type}
        
        print("\nNORMALITY TESTS:")
        print("-" * 40)
        
        try:
            jb_result = jarque_bera(series)
            jb_test = {'p_value': jb_result[1], 'is_normal': jb_result[1] > 0.05}
            print(f"Jarque-Bera: p={jb_test['p_value']:.6f}, normal={'YES' if jb_test['is_normal'] else 'NO'}")
        except:
            jb_test = {'is_normal': False}
        
        try:
            ad_result = anderson(series)
            ad_test = {'is_normal': ad_result[0] < ad_result[1][2]}
            print(f"Anderson-Darling: normal={'YES' if ad_test['is_normal'] else 'NO'} (5%)")
        except:
            ad_test = {'is_normal': False}
        
        analysis_results['distribution_tests'] = {'jarque_bera': jb_test, 'anderson_darling': ad_test}
        
        print("\nAUTOCORRELATION:")
        print("-" * 40)
        
        max_lag = min(50, len(series) // 4)
        acf_values = acf(series, nlags=max_lag, fft=True)
        pacf_values = pacf(series, nlags=max_lag)
        
        significant_lags_acf = []
        for lag in range(1, min(25, max_lag + 1)):
            if abs(acf_values[lag]) > 1.96 / np.sqrt(len(series)):
                significant_lags_acf.append(lag)
        
        print(f"Significant ACF lags: {significant_lags_acf[:10]}")
        analysis_results['autocorrelation'] = {'significant_acf_lags': significant_lags_acf}
        
        print("\nSEASONALITY:")
        print("-" * 40)
        
        if len(series) >= 24 * 7 * 2:
            try:
                decomposition = seasonal_decompose(series, model='additive', period=24, extrapolate_trend='freq')
                seasonal = decomposition.seasonal
                residual = decomposition.resid
                seasonal_strength = 1 - np.var(residual) / np.var(seasonal + residual) if np.var(seasonal + residual) > 0 else 0
                print(f"Seasonality strength: {seasonal_strength:.4f}")
                analysis_results['seasonality'] = {'strength': seasonal_strength}
            except:
                print("  Seasonal decomposition failed")
                analysis_results['seasonality'] = None
        else:
            print("Insufficient data")
            analysis_results['seasonality'] = None
        
        print("\nHETEROSKEDASTICITY:")
        print("-" * 40)
        
        try:
            arch_result = het_arch(series)
            arch_test = {'p_value': arch_result[1], 'is_heteroskedastic': arch_result[1] < 0.05}
            print(f"ARCH Test: p={arch_test['p_value']:.6f}, heteroskedastic={'YES' if arch_test['is_heteroskedastic'] else 'NO'}")
        except:
            arch_test = {'is_heteroskedastic': False}
        
        analysis_results['heteroskedasticity'] = {'arch_test': arch_test}
        
        print("\nOUTLIERS:")
        print("-" * 40)
        
        if len(series) >= 50:
            outlier_mask_hampel = self._hampel_filter(series, window=10, n_sigmas=3)
            hampel_count = len(series[outlier_mask_hampel])
        else:
            hampel_count = 0
        
        print(f"Hampel: {hampel_count} points ({hampel_count/len(series)*100:.2f}%)")
        analysis_results['outliers'] = {'hampel_percentage': (hampel_count / len(series)) * 100}
        
        print("\nSUMMARY:")
        print("-" * 40)
        
        has_seasonality = analysis_results['seasonality'] is not None and analysis_results['seasonality'].get('strength', 0) > 0.3
        
        summary = {
            'is_stationary': adf_test.get('is_stationary', False),
            'is_normal': jb_test.get('is_normal', False),
            'has_seasonality': has_seasonality,
            'is_heteroskedastic': arch_test.get('is_heteroskedastic', False),
            'outlier_percentage': (hampel_count / len(series)) * 100 if len(series) > 0 else 0,
            'series_type': series_type,
            'mean': basic_stats['mean'],
            'std': basic_stats['std']
        }
        
        print(f"Stationary: {'YES' if summary['is_stationary'] else 'NO'}")
        print(f"Normal: {'YES' if summary['is_normal'] else 'NO'}")
        print(f"Seasonality: {'YES' if summary['has_seasonality'] else 'NO'}")
        print(f"Heteroskedastic: {'YES' if summary['is_heteroskedastic'] else 'NO'}")
        print(f"Outliers: {summary['outlier_percentage']:.2f}%")
        
        analysis_results['summary'] = summary
        self.results[column_name] = analysis_results
        
        return analysis_results
    
    def analyze_all_columns(self):
        print("\n" + "="*70)
        print("ANALYZING ALL COUNTRIES")
        print("="*70)
        
        all_results = []
        total = len(self.data.columns)
        
        for idx, col in enumerate(self.data.columns, 1):
            print(f"\n[{idx}/{total}] {col}")
            result = self.comprehensive_analysis(col)
            
            if result:
                summary = result['summary']
                # Extract country name from column
                country = col.replace('_load', '')
                all_results.append({
                    'Country': country,
                    'Mean': round(summary['mean'], 2),
                    'Std': round(summary['std'], 2),
                    'Stationary': 'YES' if summary['is_stationary'] else 'NO',
                    'Normal': 'YES' if summary['is_normal'] else 'NO',
                    'Seasonality': 'YES' if summary['has_seasonality'] else 'NO',
                    'Heteroskedastic': 'YES' if summary.get('is_heteroskedastic', False) else 'NO',
                    'Outliers_%': round(summary['outlier_percentage'], 2)
                })
        
        if all_results:
            summary_df = pd.DataFrame(all_results)
            print("\n" + "="*70)
            print("SUMMARY OF ALL COUNTRIES")
            print("="*70)
            print(summary_df.to_string(index=False))
            summary_df.to_csv('statistical_analysis_summary.csv', index=False)
            print("\nSummary saved to: statistical_analysis_summary.csv")
            return summary_df
        return None

def load_real_data():
    # Direct path to your data
    data_dir = r'C:\Users\Zahara\Documents\Zoom\europe_energy_forecast\data'
    load_file = os.path.join(data_dir, 'monthly_hourly_load_values_2024.csv')
    
    print(f"Loading data from: {load_file}")
    
    if not os.path.exists(load_file):
        print("\n" + "="*70)
        print("ERROR: DATA FILE NOT FOUND!")
        print("="*70)
        print(f"Expected file: {load_file}")
        print("\nPlease make sure the data file exists in the correct location.")
        print("The program will now exit.")
        sys.exit(1)
    
    try:
        df_load = pd.read_csv(load_file, sep='\t')
        print(f"Loaded {df_load.shape[0]} rows, {df_load.shape[1]} columns")
        
        # Get all unique countries
        countries = df_load['CountryCode'].unique()
        print(f"\nCountries found: {sorted(countries.tolist())}")
        print(f"Total countries: {len(countries)}")
        
        # Create a dictionary for each country's load data
        data_dict = {}
        
        # Get timestamp index
        if 'utc_timestamp' in df_load.columns:
            timestamp_index = pd.to_datetime(df_load['utc_timestamp'].unique())
        else:
            timestamp_index = pd.date_range('2024-01-01', periods=len(df_load[df_load['CountryCode'] == countries[0]]), freq='h')
        
        # Extract data for each country
        for country in countries:
            country_data = df_load[df_load['CountryCode'] == country]
            if len(country_data) > 0:
                data_dict[f'{country}_load'] = country_data['Value'].values
        
        # Find minimum length
        min_len = min(len(v) for v in data_dict.values() if len(v) > 0)
        
        # Trim all to same length
        for key in data_dict:
            data_dict[key] = data_dict[key][:min_len]
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        df.index = timestamp_index[:min_len]
        
        print(f"\nCreated DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Countries: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("The program will now exit.")
        sys.exit(1)

def main():
    print("=" * 70)
    print("STATISTICAL ANALYSIS - ALL COUNTRIES (2024 DATA)")
    print("=" * 70)
    
    print("\nACADEMIC STANDARDS APPLIED:")
    print("   1. Hyndman seasonal strength formula")
    print("   2. Proper series classification (I(0)/I(1)/trend-stationary)")
    print("   3. Hampel filter for outlier detection")
    print("   4. Multiple normality tests (JB, AD)")
    print("   5. ARCH test for heteroskedasticity")
    print("=" * 70)
    
    # Load real data only
    df = load_real_data()
    
    if df is not None:
        print(f"\nData loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Create analyzer
        analyzer = StatisticalAnalyzer(df)
        
        # Analyze all columns (all countries)
        analyzer.analyze_all_columns()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nResults saved to: statistical_analysis_summary.csv")

if __name__ == "__main__":
    main()
