#hybrid_ensemble
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

import os
import time
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class HybridEnsembleForecaster:
    """Hybrid ensemble forecaster combining multiple ML models"""
    
    def __init__(self, data_path='data/europe_energy_real.csv'):
        self.data_path = data_path
        self.results = {}
        self.country_stats = {}
        
        print(f"\n{'='*80}")
        print("MULTI-COUNTRY ENERGY LOAD FORECASTING - HYBRID ENSEMBLE")
        print(f"{'='*80}")
    
    def detect_countries(self, n_samples=10000):
        """Detect all available countries in the dataset"""
        print("1. Detecting available countries...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path, nrows=n_samples)
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        pattern = r'([A-Z]{2})_load_actual_entsoe_transparency'
        
        countries = []
        for col in df.columns:
            match = re.match(pattern, col)
            if match:
                country_code = match.group(1)
                if country_code not in countries:
                    countries.append(country_code)
        
        print(f"   Found {len(countries)} countries: {', '.join(countries)}")
        return sorted(countries)
    
    def analyze_country(self, country, n_samples=30000):
        """Analyze a single country with HYBRID ENSEMBLE"""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {country} (HYBRID ENSEMBLE)")
        print(f"{'='*60}")
        
        try:
            df = pd.read_csv(self.data_path, nrows=n_samples)
            df.columns = [col.strip().replace(' ', '_') for col in df.columns]
            
            target_col = f"{country}_load_actual_entsoe_transparency"
            if target_col not in df.columns:
                print(f"   Skipping {country}: Target column not found")
                return None
            
            X, y, lag_1_values = self._extract_features_clean(df, country)
            
            if len(y) < 1000:
                print(f"   Skipping {country}: Insufficient data ({len(y)} samples)")
                return None
            
            split_idx = int(len(X) * 0.8)
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            lag_1_test = lag_1_values[split_idx:] if lag_1_values is not None else None
            
            scaler_linear = StandardScaler()
            X_train_scaled = scaler_linear.fit_transform(X_train)
            X_test_scaled = scaler_linear.transform(X_test)
            
            # Train models and get predictions for ensemble
            models, predictions_train, predictions_test = self._train_ensemble_models(
                X_train, X_test, X_train_scaled, X_test_scaled, y_train
            )
            
            if not predictions_test:
                print(f"   Skipping {country}: No models trained successfully")
                return None
            
            # HYBRID ENSEMBLE: Weighted average of all model predictions
            ensemble_pred_train, ensemble_pred_test, weights = self._weighted_ensemble(
                predictions_train, predictions_test, y_train
            )
            
            # Calculate ensemble metrics
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred_test))
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred_test)
            ensemble_r2 = r2_score(y_test, ensemble_pred_test)
            
            # Calculate individual model metrics
            models_results = {}
            for name, pred in predictions_test.items():
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                mae = mean_absolute_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                models_results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            
            # Add ensemble to results
            models_results['Hybrid_Ensemble'] = {
                'rmse': ensemble_rmse,
                'mae': ensemble_mae,
                'r2': ensemble_r2
            }
            
            # Calculate baseline
            baseline_metrics = None
            if lag_1_test is not None and len(lag_1_test) == len(y_test):
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
                [(name, m['rmse']) for name, m in models_results.items() if name != 'Hybrid_Ensemble'],
                key=lambda x: x[1]
            )[0]
            
            # Store results
            country_result = {
                'country': country,
                'samples': len(y),
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'best_individual_model': best_model_name,
                'ensemble_model': 'Hybrid_Ensemble',
                'ensemble_rmse': ensemble_rmse,
                'ensemble_mae': ensemble_mae,
                'ensemble_r2': ensemble_r2,
                'ensemble_weights': weights,
                'baseline_rmse': baseline_metrics['rmse'] if baseline_metrics else None,
                'baseline_r2': baseline_metrics['r2'] if baseline_metrics else None,
                'all_models': models_results,
                'load_mean': y.mean(),
                'load_std': y.std(),
                'load_min': y.min(),
                'load_max': y.max()
            }
            
            if baseline_metrics:
                rmse_improvement = ((baseline_metrics['rmse'] - ensemble_rmse) / 
                                   baseline_metrics['rmse']) * 100
                country_result['improvement_pct'] = rmse_improvement
            
            print(f"\n   Results for {country} (HYBRID ENSEMBLE):")
            print(f"     Samples: {len(y):,}")
            print(f"     Ensemble R²: {ensemble_r2:.4f}")
            print(f"     Ensemble RMSE: {ensemble_rmse:.1f} MW")
            print(f"     Best Individual: {best_model_name}")
            print(f"     Ensemble Weights: {weights}")
            if baseline_metrics:
                print(f"     Baseline R²: {baseline_metrics['r2']:.4f}")
                print(f"     Improvement: {rmse_improvement:.1f}%")
            
            return country_result
            
        except Exception as e:
            print(f"   Error analyzing {country}: {str(e)[:100]}")
            return None
    
    def _extract_features_clean(self, df, country):
        """Extract features WITHOUT LEAKAGE"""
        target_col = f"{country}_load_actual_entsoe_transparency"
        df['load'] = df[target_col].copy()
        
        features = {}
        
        df_baseline = df.copy()
        
        lags = [1, 2, 3, 24, 48, 168]
        for lag in lags:
            if lag < len(df):
                col_name = f'lag_{lag}'
                df[col_name] = df['load'].shift(lag)
                df[col_name] = df[col_name].ffill()
                features[col_name] = df[col_name].values
        
        df_baseline['lag_1'] = df_baseline['load'].shift(1)
        df_baseline['lag_1'] = df_baseline['lag_1'].ffill()
        lag_1_values = df_baseline['lag_1'].values
        
        if 'utc_timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['utc_timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                
                time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
                for feat in time_features:
                    features[feat] = df[feat].values
            except:
                pass
        
        feature_names = list(features.keys())
        X = np.column_stack([features[name] for name in feature_names])
        y = df['load'].values
        
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(lag_1_values)
        
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        lag_1_values_clean = lag_1_values[valid_mask]
        
        return X_clean, y_clean, lag_1_values_clean
    
    def _train_ensemble_models(self, X_train, X_test, X_train_scaled, X_test_scaled, y_train):
        """Train all models and return their predictions"""
        models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                n_jobs=1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=1
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'Ridge': Ridge(random_state=42)
        }
        
        model_objects = {}
        predictions_train = {}
        predictions_test = {}
        
        for name, model in models.items():
            try:
                if name == 'Ridge':
                    model.fit(X_train_scaled, y_train)
                    pred_train = model.predict(X_train_scaled)
                    pred_test = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    pred_train = model.predict(X_train)
                    pred_test = model.predict(X_test)
                
                model_objects[name] = model
                predictions_train[name] = pred_train
                predictions_test[name] = pred_test
                
            except Exception as e:
                print(f"     Warning: {name} failed: {str(e)[:50]}")
                continue
        
        return model_objects, predictions_train, predictions_test
    
    def _weighted_ensemble(self, predictions_train, predictions_test, y_train):
        """Calculate optimal weights for weighted average ensemble"""
        model_names = list(predictions_train.keys())
        n_models = len(model_names)
        
        if n_models == 0:
            return None, None, None
        
        if n_models == 1:
            return predictions_train[model_names[0]], predictions_test[model_names[0]], {model_names[0]: 1.0}
        
        predictions_matrix = np.column_stack([predictions_train[name] for name in model_names])
        
        try:
            from sklearn.linear_model import Ridge
            meta_learner = Ridge(alpha=1.0, random_state=42)
            meta_learner.fit(predictions_matrix, y_train)
            weights = meta_learner.coef_
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
        except:
            correlation_weights = []
            for name in model_names:
                corr = np.corrcoef(y_train, predictions_train[name])[0, 1]
                correlation_weights.append(max(0, corr))
            weights = np.array(correlation_weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n_models) / n_models
        
        ensemble_train = np.zeros(len(y_train))
        ensemble_test = np.zeros(len(list(predictions_test.values())[0]))
        
        for i, name in enumerate(model_names):
            ensemble_train += weights[i] * predictions_train[name]
            ensemble_test += weights[i] * predictions_test[name]
        
        weight_dict = {name: float(weights[i]) for i, name in enumerate(model_names)}
        
        return ensemble_train, ensemble_test, weight_dict
    
    def run_all_countries(self, max_countries=None, n_samples=30000):
        """Run analysis for all detected countries"""
        countries = self.detect_countries()
        
        if max_countries:
            countries = countries[:max_countries]
            print(f"   Analyzing first {max_countries} countries")
        
        print(f"\n2. Analyzing {len(countries)} countries (HYBRID ENSEMBLE)...")
        print(f"{'='*60}")
        
        for country in tqdm(countries, desc="Countries (Hybrid Ensemble)"):
            result = self.analyze_country(country, n_samples)
            if result:
                self.results[country] = result
        
        print(f"\n{'='*80}")
        print(f"COMPLETED: Analyzed {len(self.results)} countries successfully (HYBRID ENSEMBLE)")
        print(f"{'='*80}")
        
        return self.results
    
    def generate_comparative_report(self):
        """Generate comparative report across countries"""
        if not self.results:
            print("No results to report")
            return None
        
        print(f"\n{'='*80}")
        print("COMPARATIVE ANALYSIS REPORT (HYBRID ENSEMBLE)")
        print(f"{'='*80}")
        
        data = []
        for country, result in self.results.items():
            data.append({
                'Country': country,
                'Samples': result['samples'],
                'Load Mean (MW)': int(result['load_mean']),
                'Load Std (MW)': int(result['load_std']),
                'Ensemble R²': result['ensemble_r2'],
                'Ensemble RMSE (MW)': int(result['ensemble_rmse']),
                'Best Individual': result['best_individual_model'],
                'Baseline R²': result['baseline_r2'],
                'Improvement %': result.get('improvement_pct', 0)
            })
        
        df_report = pd.DataFrame(data)
        df_report = df_report.sort_values('Ensemble R²', ascending=False)
        
        print("\nCountry Performance Ranking (by Ensemble R²) - HYBRID ENSEMBLE:")
        print("-" * 100)
        print(df_report.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS (HYBRID ENSEMBLE)")
        print(f"{'='*80}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Average Ensemble R²:          {df_report['Ensemble R²'].mean():.4f}")
        print(f"  Median Ensemble R²:           {df_report['Ensemble R²'].median():.4f}")
        print(f"  Best Ensemble R²:             {df_report['Ensemble R²'].max():.4f} ({df_report.iloc[0]['Country']})")
        print(f"  Average Improvement:           {df_report['Improvement %'].mean():.1f}%")
        
        return df_report
    
    def save_detailed_report(self, filename='multi_country_results_HYBRID.csv'):
        """Save detailed results to CSV"""
        if not self.results:
            print("No results to save")
            return
        
        detailed_data = []
        for country, result in self.results.items():
            row = {
                'country': country,
                'total_samples': result['samples'],
                'train_samples': result['train_samples'],
                'test_samples': result['test_samples'],
                'load_mean_mw': result['load_mean'],
                'load_std_mw': result['load_std'],
                'best_individual_model': result['best_individual_model'],
                'ensemble_r2': result['ensemble_r2'],
                'ensemble_rmse_mw': result['ensemble_rmse'],
                'ensemble_mae_mw': result['ensemble_mae'],
                'ensemble_weights': str(result['ensemble_weights']),
                'baseline_r2': result['baseline_r2'],
                'baseline_rmse_mw': result['baseline_rmse'],
                'improvement_pct': result.get('improvement_pct', None),
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            for model_name, metrics in result['all_models'].items():
                row[f'{model_name}_r2'] = metrics['r2']
                row[f'{model_name}_rmse'] = metrics['rmse']
            
            detailed_data.append(row)
        
        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"\nDetailed HYBRID ENSEMBLE results saved to {filename}")
        print(f"Total countries analyzed: {len(self.results)}")
        
        return df_detailed


def main():
    """Main function for hybrid ensemble analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Country Energy Load Forecasting (HYBRID ENSEMBLE)')
    parser.add_argument('--max-countries', type=int, default=None,
                       help='Maximum number of countries to analyze')
    parser.add_argument('--samples', type=int, default=30000,
                       help='Number of samples per country')
    parser.add_argument('--output', type=str, default='multi_country_results_HYBRID.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("EUROPEAN ENERGY LOAD FORECASTING - HYBRID ENSEMBLE ANALYSIS")
    print(f"{'='*80}")
    
    forecaster = HybridEnsembleForecaster(
        data_path='data/europe_energy_real.csv'
    )
    
    results = forecaster.run_all_countries(
        max_countries=args.max_countries,
        n_samples=args.samples
    )
    
    if results:
        report_df = forecaster.generate_comparative_report()
        forecaster.save_detailed_report(args.output)
        
        print(f"\n{'='*80}")
        print("HYBRID ENSEMBLE ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"Successfully analyzed {len(results)} countries")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    main()
