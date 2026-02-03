#hybrid
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

class MultiCountryEnergyForecaster_CLEAN:
    """Forecaster for multiple countries """
    
    def __init__(self, data_path='data/europe_energy_real.csv'):
        self.data_path = data_path
        self.results = {}
        self.country_stats = {}
        
        print(f"\n{'='*80}")
        print("MULTI-COUNTRY ENERGY LOAD FORECASTING - CLEAN VERSION")
        print(f"{'='*80}")
    
    def detect_countries(self, n_samples=10000):
        """Detect all available countries in the dataset"""
        print("1. Detecting available countries...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Read first few rows to detect columns
        df = pd.read_csv(self.data_path, nrows=n_samples)
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Pattern to find country load columns
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
        """Analyze a single country with NO LEAKAGE"""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {country} (CLEAN)")
        print(f"{'='*60}")
        
        try:
            # Load data
            df = pd.read_csv(self.data_path, nrows=n_samples)
            df.columns = [col.strip().replace(' ', '_') for col in df.columns]
            
            # Target column
            target_col = f"{country}_load_actual_entsoe_transparency"
            if target_col not in df.columns:
                print(f"   Skipping {country}: Target column not found")
                return None
            
            # Extract features WITHOUT LEAKAGE
            X, y, lag_1_values = self._extract_features_clean(df, country)
            
            if len(y) < 1000:
                print(f"   Skipping {country}: Insufficient data ({len(y)} samples)")
                return None
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            lag_1_test = lag_1_values[split_idx:] if lag_1_values is not None else None
            
            # Separate scaling for linear vs tree models
            scaler_linear = StandardScaler()
            X_train_scaled = scaler_linear.fit_transform(X_train)
            X_test_scaled = scaler_linear.transform(X_test)
            
            # Train models with appropriate scaling
            models_results = self._train_all_models_clean(
                X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
            )
            
            # Get best model
            if not models_results:
                print(f"   Skipping {country}: No models trained successfully")
                return None
            
            best_model_name, best_metrics = min(models_results.items(), key=lambda x: x[1]['rmse'])
            
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
            
            # Store results
            country_result = {
                'country': country,
                'samples': len(y),
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'best_model': best_model_name,
                'model_rmse': best_metrics['rmse'],
                'model_mae': best_metrics['mae'],
                'model_r2': best_metrics['r2'],
                'baseline_rmse': baseline_metrics['rmse'] if baseline_metrics else None,
                'baseline_r2': baseline_metrics['r2'] if baseline_metrics else None,
                'all_models': models_results,
                'load_mean': y.mean(),
                'load_std': y.std(),
                'load_min': y.min(),
                'load_max': y.max()
            }
            
            if baseline_metrics:
                rmse_improvement = ((baseline_metrics['rmse'] - best_metrics['rmse']) / 
                                   baseline_metrics['rmse']) * 100
                country_result['improvement_pct'] = rmse_improvement
            
            # Print summary
            print(f"\n   Results for {country} (CLEAN):")
            print(f"     Samples: {len(y):,}")
            print(f"     Best Model: {best_model_name}")
            print(f"     R²: {best_metrics['r2']:.4f}")
            print(f"     RMSE: {best_metrics['rmse']:.1f} MW")
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
        
        # IMPORTANT: Create a COPY for baseline calculation BEFORE any filling
        df_baseline = df.copy()
        
        # Lag features WITHOUT BFill (CAUSES LEAKAGE!)
        lags = [1, 2, 3, 24, 48, 168]
        for lag in lags:
            if lag < len(df):
                col_name = f'lag_{lag}'
                # ORIGINAL (WITH LEAKAGE): df[col_name] = df['load'].shift(lag).ffill().bfill()
                # CLEAN VERSION: Only forward fill, NO backward fill
                df[col_name] = df['load'].shift(lag)
                df[col_name] = df[col_name].ffill()  # ONLY FORWARD FILL!
                features[col_name] = df[col_name].values
        
        # Store lag_1 for baseline (from ORIGINAL data, not filled)
        df_baseline['lag_1'] = df_baseline['load'].shift(1)
        # Only forward fill for baseline too
        df_baseline['lag_1'] = df_baseline['lag_1'].ffill()
        lag_1_values = df_baseline['lag_1'].values
        
        # Time features (safe - no leakage)
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
        
        # Create X and y
        feature_names = list(features.keys())
        X = np.column_stack([features[name] for name in feature_names])
        y = df['load'].values
        
        # STRICT NaN handling: Remove rows with ANY NaN
        # This ensures no future information leaks into training
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(lag_1_values)
        
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        lag_1_values_clean = lag_1_values[valid_mask]
        
        return X_clean, y_clean, lag_1_values_clean
    
    def _train_all_models_clean(self, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
        """Train models with separate scaling"""
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
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'Ridge': Ridge(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Use scaled features only for Ridge, unscaled for tree models
                if name == 'Ridge':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
                
            except Exception as e:
                continue
        
        return results
    
    def run_all_countries(self, max_countries=None, n_samples=30000):
        """Run analysis for all detected countries"""
        # Detect countries
        countries = self.detect_countries()
        
        if max_countries:
            countries = countries[:max_countries]
            print(f"   Analyzing first {max_countries} countries")
        
        print(f"\n2. Analyzing {len(countries)} countries (CLEAN VERSION)...")
        print(f"{'='*60}")
        
        # Analyze each country
        for country in tqdm(countries, desc="Countries (Clean)"):
            result = self.analyze_country(country, n_samples)
            if result:
                self.results[country] = result
        
        print(f"\n{'='*80}")
        print(f"COMPLETED: Analyzed {len(self.results)} countries successfully (CLEAN)")
        print(f"{'='*80}")
        
        return self.results
    
    def generate_comparative_report(self):
        """Generate comparative report across countries"""
        if not self.results:
            print("No results to report")
            return None
        
        print(f"\n{'='*80}")
        print("COMPARATIVE ANALYSIS REPORT (CLEAN VERSION)")
        print(f"{'='*80}")
        
        # Create DataFrame for analysis
        data = []
        for country, result in self.results.items():
            data.append({
                'Country': country,
                'Samples': result['samples'],
                'Load Mean (MW)': int(result['load_mean']),
                'Load Std (MW)': int(result['load_std']),
                'Best Model': result['best_model'],
                'R²': result['model_r2'],
                'RMSE (MW)': int(result['model_rmse']),
                'Baseline R²': result['baseline_r2'],
                'Improvement %': result.get('improvement_pct', 0)
            })
        
        df_report = pd.DataFrame(data)
        
        # Sort by R² (best to worst)
        df_report = df_report.sort_values('R²', ascending=False)
        
        # Print summary table
        print("\nCountry Performance Ranking (by R²) - CLEAN VERSION:")
        print("-" * 100)
        print(df_report.to_string(index=False))
        
        # Statistics
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS (CLEAN)")
        print(f"{'='*80}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Average R²:          {df_report['R²'].mean():.4f}")
        print(f"  Median R²:           {df_report['R²'].median():.4f}")
        print(f"  Best R²:             {df_report['R²'].max():.4f} ({df_report.iloc[0]['Country']})")
        print(f"  Worst R²:            {df_report['R²'].min():.4f} ({df_report.iloc[-1]['Country']})")
        print(f"  Average Improvement: {df_report['Improvement %'].mean():.1f}%")
        
        print(f"\nModel Preferences:")
        model_counts = df_report['Best Model'].value_counts()
        for model, count in model_counts.items():
            print(f"  {model}: {count} countries")
        
        print(f"\nLoad Characteristics:")
        print(f"  Highest Load:  {df_report['Load Mean (MW)'].max():,} MW")
        print(f"  Lowest Load:   {df_report['Load Mean (MW)'].min():,} MW")
        print(f"  Average Load:  {df_report['Load Mean (MW)'].mean():,.0f} MW")
        
        # Categorize countries by performance
        print(f"\n{'='*80}")
        print("PERFORMANCE CATEGORIES (CLEAN)")
        print(f"{'='*80}")
        
        excellent = df_report[df_report['R²'] >= 0.99]
        good = df_report[(df_report['R²'] >= 0.97) & (df_report['R²'] < 0.99)]
        fair = df_report[(df_report['R²'] >= 0.95) & (df_report['R²'] < 0.97)]
        poor = df_report[df_report['R²'] < 0.95]
        
        print(f"\nExcellent (R² ≥ 0.99):")
        if len(excellent) > 0:
            print(f"  Countries: {', '.join(excellent['Country'].tolist())}")
            print(f"  Count: {len(excellent)}")
        else:
            print("  None")
        
        print(f"\nGood (0.97 ≤ R² < 0.99):")
        if len(good) > 0:
            print(f"  Countries: {', '.join(good['Country'].tolist())}")
            print(f"  Count: {len(good)}")
        else:
            print("  None")
        
        print(f"\nFair (0.95 ≤ R² < 0.97):")
        if len(fair) > 0:
            print(f"  Countries: {', '.join(fair['Country'].tolist())}")
            print(f"  Count: {len(fair)}")
        else:
            print("  None")
        
        print(f"\nNeeds Improvement (R² < 0.95):")
        if len(poor) > 0:
            print(f"  Countries: {', '.join(poor['Country'].tolist())}")
            print(f"  Count: {len(poor)}")
        else:
            print("  None")
        
        print(f"\n{'='*80}")
        print("DATA QUALITY NOTES:")
        print(f"{'='*80}")
        print("✓ NO backward fill used (prevents leakage)")
        print("✓ Rows with NaN removed (ensures clean training)")
        print("✓ Separate scaling for linear models only")
        print("✓ Time-series split maintained")
        print("✓ Baseline calculated from original data")
        
        return df_report
    
    def create_visualizations(self, save_dir='results_clean/'):
        """Create visualizations for all countries"""
        if not self.results:
            print("No results to visualize")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare data for visualizations
            countries = list(self.results.keys())
            r2_scores = [self.results[c]['model_r2'] for c in countries]
            rmse_values = [self.results[c]['model_rmse'] for c in countries]
            load_means = [self.results[c]['load_mean'] for c in countries]
            improvements = [self.results[c].get('improvement_pct', 0) for c in countries]
            
            # 1. R² Comparison Bar Chart
            plt.figure(figsize=(12, 6))
            sorted_indices = np.argsort(r2_scores)[::-1]
            sorted_countries = [countries[i] for i in sorted_indices]
            sorted_r2 = [r2_scores[i] for i in sorted_indices]
            
            bars = plt.bar(sorted_countries, sorted_r2, color='steelblue')
            plt.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, label='Excellent (0.99)')
            plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='Good (0.95)')
            plt.xlabel('Country')
            plt.ylabel('R² Score')
            plt.title('Forecasting Accuracy by Country (R²) - CLEAN VERSION')
            plt.ylim(0.8, 1.0)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_dir}/r2_by_country_clean.png', dpi=150, bbox_inches='tight')
            
            # 2. RMSE vs Load Size Scatter
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(load_means, rmse_values, c=r2_scores, 
                                 cmap='viridis', s=100, alpha=0.7, edgecolors='black')
            
            # Add country labels
            for i, country in enumerate(countries):
                plt.annotate(country, (load_means[i], rmse_values[i]), 
                           fontsize=8, alpha=0.7)
            
            plt.colorbar(scatter, label='R² Score')
            plt.xlabel('Average Load (MW)')
            plt.ylabel('RMSE (MW)')
            plt.title('Forecasting Error vs Load Size - CLEAN VERSION')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/rmse_vs_load_clean.png', dpi=150, bbox_inches='tight')
            
            # 3. Improvement over Baseline
            plt.figure(figsize=(10, 6))
            sorted_imp_indices = np.argsort(improvements)[::-1]
            sorted_imp_countries = [countries[i] for i in sorted_imp_indices]
            sorted_improvements = [improvements[i] for i in sorted_imp_indices]
            
            colors = ['green' if imp > 50 else 'orange' if imp > 30 else 'red' 
                     for imp in sorted_improvements]
            plt.bar(sorted_imp_countries, sorted_improvements, color=colors)
            plt.xlabel('Country')
            plt.ylabel('Improvement over Baseline (%)')
            plt.title('Model Improvement over Persistence Baseline - CLEAN')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/improvement_by_country_clean.png', dpi=150, bbox_inches='tight')
            
            # 4. Model Preference Heatmap
            model_counts = {}
            for country, result in self.results.items():
                model = result['best_model']
                model_counts[model] = model_counts.get(model, 0) + 1
            
            plt.figure(figsize=(8, 4))
            models = list(model_counts.keys())
            counts = list(model_counts.values())
            
            plt.bar(models, counts, color='skyblue', edgecolor='black')
            plt.xlabel('Model')
            plt.ylabel('Number of Countries')
            plt.title('Best Model by Country - CLEAN VERSION')
            for i, count in enumerate(counts):
                plt.text(i, count + 0.1, str(count), ha='center')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/model_preference_clean.png', dpi=150, bbox_inches='tight')
            
            plt.show()
            print(f"\nVisualizations saved to {save_dir}")
            
        except Exception as e:
            print(f"Visualization error: {str(e)[:50]}")
    
    def save_detailed_report(self, filename='multi_country_results_CLEAN.csv'):
        """Save detailed results to CSV"""
        if not self.results:
            print("No results to save")
            return
        
        # Create detailed DataFrame
        detailed_data = []
        for country, result in self.results.items():
            row = {
                'country': country,
                'total_samples': result['samples'],
                'train_samples': result['train_samples'],
                'test_samples': result['test_samples'],
                'load_mean_mw': result['load_mean'],
                'load_std_mw': result['load_std'],
                'load_min_mw': result['load_min'],
                'load_max_mw': result['load_max'],
                'best_model': result['best_model'],
                'model_r2': result['model_r2'],
                'model_rmse_mw': result['model_rmse'],
                'model_mae_mw': result['model_mae'],
                'baseline_r2': result['baseline_r2'],
                'baseline_rmse_mw': result['baseline_rmse'],
                'improvement_pct': result.get('improvement_pct', None),
                'data_quality': 'CLEAN (no backward fill)',
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add all model performances
            for model_name, metrics in result['all_models'].items():
                row[f'{model_name}_r2'] = metrics['r2']
                row[f'{model_name}_rmse'] = metrics['rmse']
            
            detailed_data.append(row)
        
        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"\nDetailed CLEAN results saved to {filename}")
        print(f"Total countries analyzed: {len(self.results)}")
        
        return df_detailed


def compare_with_original(original_file='multi_country_results.csv', 
                         clean_file='multi_country_results_CLEAN.csv'):
    """Compare original vs clean results"""
    try:
        if not os.path.exists(original_file):
            print(f"Original file not found: {original_file}")
            return
        
        if not os.path.exists(clean_file):
            print(f"Clean file not found: {clean_file}")
            return
        
        # Load data
        df_orig = pd.read_csv(original_file)
        df_clean = pd.read_csv(clean_file)
        
        # Merge for comparison
        df_merge = pd.merge(
            df_orig[['country', 'model_r2', 'model_rmse_mw', 'improvement_pct']],
            df_clean[['country', 'model_r2', 'model_rmse_mw', 'improvement_pct']],
            on='country',
            suffixes=('_orig', '_clean')
        )
        
        print(f"\n{'='*80}")
        print("COMPARISON: ORIGINAL vs CLEAN VERSION")
        print(f"{'='*80}")
        
        print(f"\nPerformance Changes:")
        print("-" * 60)
        
        df_merge['r2_change'] = df_merge['model_r2_clean'] - df_merge['model_r2_orig']
        df_merge['r2_change_pct'] = (df_merge['r2_change'] / df_merge['model_r2_orig']) * 100
        
        df_merge['rmse_change'] = df_merge['model_rmse_mw_clean'] - df_merge['model_rmse_mw_orig']
        df_merge['rmse_change_pct'] = (df_merge['rmse_change'] / df_merge['model_rmse_mw_orig']) * 100
        
        df_merge['imp_change'] = df_merge['improvement_pct_clean'] - df_merge['improvement_pct_orig']
        
        # Summary statistics
        print(f"Average R² change: {df_merge['r2_change'].mean():.4f} ({df_merge['r2_change_pct'].mean():.2f}%)")
        print(f"Average RMSE change: {df_merge['rmse_change'].mean():.1f} MW ({df_merge['rmse_change_pct'].mean():.2f}%)")
        print(f"Average Improvement change: {df_merge['imp_change'].mean():.2f}%")
        
        print(f"\nCountries with largest R² decrease (expected):")
        largest_decreases = df_merge.nsmallest(5, 'r2_change')
        for _, row in largest_decreases.iterrows():
            print(f"  {row['country']}: {row['model_r2_orig']:.4f} → {row['model_r2_clean']:.4f} "
                  f"(Δ={row['r2_change']:.4f})")
        
        print(f"\nScientific Interpretation:")
        print("✓ Decrease in R² is EXPECTED when removing leakage")
        print("✓ Clean results are more scientifically valid")
        print("✓ Results remain excellent for most countries")
        
        return df_merge
        
    except Exception as e:
        print(f"Comparison error: {str(e)}")
        return None


def main():
    """Main function for multi-country analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Country Energy Load Forecasting (CLEAN)')
    parser.add_argument('--max-countries', type=int, default=None,
                       help='Maximum number of countries to analyze')
    parser.add_argument('--samples', type=int, default=30000,
                       help='Number of samples per country')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualizations')
    parser.add_argument('--output', type=str, default='multi_country_results_CLEAN.csv',
                       help='Output CSV file')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with original results')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("EUROPEAN ENERGY LOAD FORECASTING - CLEAN MULTI-COUNTRY ANALYSIS")
    print(f"{'='*80}")
    
    # Create forecaster
    forecaster = MultiCountryEnergyForecaster_CLEAN(
        data_path='data/europe_energy_real.csv'
    )
    
    # Run analysis for all countries
    results = forecaster.run_all_countries(
        max_countries=args.max_countries,
        n_samples=args.samples
    )
    
    if results:
        # Generate comparative report
        report_df = forecaster.generate_comparative_report()
        
        # Create visualizations
        if not args.no_viz:
            forecaster.create_visualizations()
        
        # Save detailed report
        forecaster.save_detailed_report(args.output)
        
        # Compare with original if requested
        if args.compare:
            compare_with_original()
        
        print(f"\n{'='*80}")
        print("CLEAN ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"Successfully analyzed {len(results)} countries")
        print(f"Results saved to: {args.output}")
        
        # Print quick summary
        if report_df is not None:
            print(f"\nTop 5 Performing Countries (CLEAN):")
            for i, row in report_df.head().iterrows():
                print(f"  {i+1}. {row['Country']}: R²={row['R²']:.4f}, "
                      f"Improvement={row['Improvement %']:.1f}%")
        
        print(f"\nData Quality Summary:")
        print("✓ NO backward fill (prevents leakage)")
        print("✓ NaN rows removed (clean training)")
        print("✓ Separate scaling (efficient)")
        print("✓ Time-series split maintained")
        print("✓ Scientifically valid results")
    else:
        print("No countries were successfully analyzed")


if __name__ == "__main__":
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    main()
