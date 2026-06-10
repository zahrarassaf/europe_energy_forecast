import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CARBON IMPACT ANALYZER
# ============================================================================

class CarbonImpactAnalyzer:
    def __init__(self):
        self.co2_intensity_by_country = {
            'AT': 103.48, 'BE': 126.96, 'BG': 278.85, 'CH': 34.58, 'CY': 511.23,
            'CZ': 414.23, 'DE': 336.38, 'DK': 131.77, 'EE': 343.45, 'ES': 146.22,
            'FI': 66.63, 'FR': 40.48, 'GB': 216.50, 'GR': 321.65, 'HR': 170.70,
            'HU': 184.47, 'IE': 270.91, 'IT': 281.40, 'LT': 116.37, 'LU': 132.45,
            'LV': 134.28, 'ME': 422.10, 'NL': 250.72, 'NO': 29.66, 'PL': 608.18,
            'PT': 110.64, 'RO': 251.33, 'RS': 666.40, 'SE': 34.91, 'SI': 230.40,
            'SK': 96.55, 'UA': 250.47
        }
    
    def calculate_carbon_reduction(self, avg_consumption, improvement, country_code='AT'):
        if improvement <= 0:
            return {
                'annual_co2_reduction_tons': 0.0,
                'equivalent_cars_removed': 0,
                'equivalent_trees_planted': 0,
                'annual_energy_savings_mwh': 0.0
            }
        
        avg_co2 = self.co2_intensity_by_country.get(country_code, 300)
        hours_per_year = 8760
        
        annual_energy_savings = avg_consumption * improvement * hours_per_year
        annual_co2_reduction = (annual_energy_savings * avg_co2 * 1000) / 1000000
        
        return {
            'annual_co2_reduction_tons': float(annual_co2_reduction),
            'equivalent_cars_removed': int(annual_co2_reduction / 4.6),
            'equivalent_trees_planted': int(annual_co2_reduction * 20),
            'annual_energy_savings_mwh': float(annual_energy_savings)
        }


# ============================================================================
# ECONOMIC ANALYZER
# ============================================================================

class EconomicAnalyzer:
    def __init__(self):
        self.carbon_price = 80
        self.discount_rate = 0.05
    
    def calculate_economic_savings(self, energy_savings_mwh, co2_reduction):
        avg_price = 80
        
        savings_from_efficiency = energy_savings_mwh * avg_price
        savings_from_carbon = co2_reduction * self.carbon_price
        total_annual_savings = savings_from_efficiency + savings_from_carbon
        
        initial_investment = energy_savings_mwh * 500
        
        if total_annual_savings > 0 and initial_investment > 0:
            payback_period = initial_investment / total_annual_savings
            roi_percentage = (total_annual_savings / initial_investment) * 100
            npv = total_annual_savings * ((1 - (1 + self.discount_rate)**-20) / self.discount_rate) - initial_investment
        else:
            payback_period = 999.0
            roi_percentage = 0.0
            npv = -initial_investment
        
        return {
            'total_annual_savings_eur': round(float(total_annual_savings), 0),
            'savings_from_efficiency': round(float(savings_from_efficiency), 0),
            'savings_from_carbon': round(float(savings_from_carbon), 0),
            'payback_period_years': round(float(payback_period), 1),
            'roi_percentage': round(float(roi_percentage), 1),
            'initial_investment_eur': round(float(initial_investment), 0),
            'npv_eur': round(float(npv), 0)
        }


# ============================================================================
# HYBRID ENSEMBLE FORECASTER
# ============================================================================

class HybridEnsembleForecaster:
    def __init__(self, data_path):
        self.data_path = data_path
        self.results = {}
        self.data = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
    
    def load_data(self, n_samples=30000):
        print(f"Loading data from: {self.data_path}")
        self.data = pd.read_csv(self.data_path, nrows=n_samples)
        self.data.columns = [col.strip().replace(' ', '_') for col in self.data.columns]
        
        if 'utc_timestamp' in self.data.columns:
            self.data['utc_timestamp'] = pd.to_datetime(self.data['utc_timestamp'])
            self.data.set_index('utc_timestamp', inplace=True)
        
        print(f"Data shape: {self.data.shape}")
        return self.data
    
    def get_all_countries(self):
        pattern = r'([A-Z]{2})_load_actual_entsoe_transparency'
        countries = []
        for col in self.data.columns:
            import re
            match = re.match(pattern, col)
            if match:
                country_code = match.group(1)
                if country_code not in countries:
                    countries.append(country_code)
        return sorted(countries)
    
    def extract_features(self, df, target_col):
        df = df.copy()
        df['target'] = df[target_col]
        
        features = {}
        
        lags = [1, 2, 3, 24, 48, 168]
        for lag in lags:
            if len(df) > lag:
                col_name = f'lag_{lag}'
                df[col_name] = df['target'].shift(lag).ffill()
                features[col_name] = df[col_name].values
        
        if hasattr(df.index, 'hour'):
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            
            features['hour_sin'] = df['hour_sin'].values
            features['hour_cos'] = df['hour_cos'].values
            features['day_sin'] = df['day_sin'].values
            features['day_cos'] = df['day_cos'].values
        
        X = np.column_stack([features[name] for name in features.keys()])
        y = df['target'].values
        self.feature_names = list(features.keys())
        
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        return X[valid_mask], y[valid_mask]
    
    def train_country(self, country_code):
        print(f"\n{'='*50}")
        print(f"Training: {country_code}")
        print(f"{'='*50}")
        
        target_col = f"{country_code}_load_actual_entsoe_transparency"
        if target_col not in self.data.columns:
            print(f"  Target column not found")
            return None
        
        X, y = self.extract_features(self.data, target_col)
        
        if len(X) < 1000:
            print(f"  Insufficient data: {len(X)} samples")
            return None
        
        split_idx = int(len(X) * 0.8)
        X_train_raw = X[:split_idx]
        X_test_raw = X[split_idx:]
        y_train_raw = y[:split_idx]
        y_test_raw = y[split_idx:]
        
        X_train = self.feature_scaler.fit_transform(X_train_raw)
        X_test = self.feature_scaler.transform(X_test_raw)
        
        y_train = self.target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        y_test = self.target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
        
        models = {
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0, n_jobs=1),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            'Ridge': Ridge(random_state=42)
        }
        
        predictions_test = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                predictions_test[name] = pred
            except Exception as e:
                print(f"  {name} failed: {e}")
                continue
        
        if not predictions_test:
            return None
        
        ensemble_pred = np.mean(list(predictions_test.values()), axis=0)
        
        y_test_orig = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        ensemble_orig = self.target_scaler.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
        
        persistence_pred = X_test_raw[:, 0]
        persistence_orig = self.target_scaler.inverse_transform(persistence_pred.reshape(-1, 1)).flatten()
        min_len = min(len(y_test_orig), len(persistence_orig))
        
        rmse_model = np.sqrt(mean_squared_error(y_test_orig[:min_len], ensemble_orig[:min_len]))
        rmse_persistence = np.sqrt(mean_squared_error(y_test_orig[:min_len], persistence_orig[:min_len]))
        r2 = r2_score(y_test_orig[:min_len], ensemble_orig[:min_len])
        improvement = ((rmse_persistence - rmse_model) / rmse_persistence) * 100
        
        result = {
            'country': country_code,
            'rmse': rmse_model,
            'r2': r2,
            'improvement_pct': improvement,
            'load_mean_mw': np.mean(y_test_orig),
            'samples': len(y)
        }
        
        print(f"  RMSE: {rmse_model:.1f} MW")
        print(f"  R²: {r2:.4f}")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Mean Load: {result['load_mean_mw']:.0f} MW")
        
        return result
    
    def run_all_countries(self):
        countries = self.get_all_countries()
        print(f"\nFound {len(countries)} countries")
        
        for country in countries:
            result = self.train_country(country)
            if result:
                self.results[country] = result
        
        return self.results


# ============================================================================
# COMPLETE ANALYSIS WITH FORECASTING
# ============================================================================

class CompleteEnergyAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.forecaster = HybridEnsembleForecaster(data_path)
        self.carbon_analyzer = CarbonImpactAnalyzer()
        self.economic_analyzer = EconomicAnalyzer()
        self.forecast_results = {}
        self.impact_results = {}
    
    def run_full_analysis(self):
        print("=" * 70)
        print("COMPLETE ENERGY ANALYSIS: FORECASTING + CARBON + ECONOMIC")
        print("=" * 70)
        
        print("\n" + "=" * 70)
        print("STEP 1: LOAD DATA")
        print("=" * 70)
        self.forecaster.load_data(n_samples=30000)
        
        print("\n" + "=" * 70)
        print("STEP 2: TRAIN HYBRID ENSEMBLE FOR ALL COUNTRIES")
        print("=" * 70)
        forecast_results = self.forecaster.run_all_countries()
        
        print("\n" + "=" * 70)
        print("STEP 3: CARBON AND ECONOMIC IMPACT ANALYSIS")
        print("=" * 70)
        
        all_impacts = []
        total_co2 = 0
        total_savings = 0
        total_investment = 0
        
        for country, result in forecast_results.items():
            improvement = result['improvement_pct'] / 100
            load_mean = result['load_mean_mw']
            
            carbon = self.carbon_analyzer.calculate_carbon_reduction(load_mean, improvement, country)
            economic = self.economic_analyzer.calculate_economic_savings(
                carbon['annual_energy_savings_mwh'],
                carbon['annual_co2_reduction_tons']
            )
            
            all_impacts.append({
                'Country': country,
                'R2': result['r2'],
                'Improvement_Pct': result['improvement_pct'],
                'RMSE_MW': result['rmse'],
                'Load_Mean_MW': load_mean,
                'CO2_Reduction_Tons': carbon['annual_co2_reduction_tons'],
                'Annual_Savings_EUR': economic['total_annual_savings_eur'],
                'Investment_EUR': economic['initial_investment_eur'],
                'Payback_Years': economic['payback_period_years'],
                'ROI_Pct': economic['roi_percentage']
            })
            
            total_co2 += carbon['annual_co2_reduction_tons']
            total_savings += economic['total_annual_savings_eur']
            total_investment += economic['initial_investment_eur']
            
            print(f"\n{country}:")
            print(f"  R²: {result['r2']:.4f}, Improvement: {result['improvement_pct']:.1f}%")
            print(f"  CO2 Reduction: {carbon['annual_co2_reduction_tons']:,.0f} tons/year")
            print(f"  Annual Savings: EUR{economic['total_annual_savings_eur']:,.0f}")
            print(f"  Payback: {economic['payback_period_years']:.1f} years")
            print(f"  ROI: {economic['roi_percentage']:.1f}%")
        
        df_impacts = pd.DataFrame(all_impacts)
        df_impacts = df_impacts.sort_values('CO2_Reduction_Tons', ascending=False)
        
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        print(f"\nTotal CO2 Reduction: {total_co2:,.0f} tons/year")
        print(f"Total Annual Savings: EUR{total_savings:,.0f}")
        print(f"Total Investment: EUR{total_investment:,.0f}")
        
        print("\n" + "=" * 70)
        print("TOP 10 COUNTRIES BY CO2 REDUCTION")
        print("=" * 70)
        print(df_impacts[['Country', 'R2', 'Improvement_Pct', 'CO2_Reduction_Tons', 'ROI_Pct', 'Payback_Years']].head(10).to_string(index=False))
        
        df_impacts.to_csv('complete_energy_analysis_31_countries.csv', index=False)
        print("\nSaved to: complete_energy_analysis_31_countries.csv")
        
        self.forecast_results = forecast_results
        self.impact_results = df_impacts
        
        return forecast_results, df_impacts
    
    def forecast_single_country(self, country_code, n_days=7):
        print(f"\n{'='*60}")
        print(f"FORECASTING FOR {country_code} ({n_days} days)")
        print(f"{'='*60}")
        
        if country_code not in self.forecast_results:
            print(f"Country {country_code} not trained yet. Running training...")
            self.forecaster.load_data(n_samples=30000)
            result = self.forecaster.train_country(country_code)
            if not result:
                print(f"Cannot forecast for {country_code}")
                return None
            self.forecast_results[country_code] = result
        
        return self.forecast_results[country_code]
    
    def generate_report(self):
        print("\n" + "=" * 70)
        print("FINAL REPORT")
        print("=" * 70)
        
        if not self.impact_results.empty:
            print("\nPERFORMANCE RANKING (by R²):")
            print(self.impact_results[['Country', 'R2', 'Improvement_Pct']].head(10).to_string(index=False))
            
            print("\nENVIRONMENTAL IMPACT RANKING (by CO2 Reduction):")
            print(self.impact_results[['Country', 'CO2_Reduction_Tons', 'ROI_Pct', 'Payback_Years']].head(10).to_string(index=False))
            
            total_co2 = self.impact_results['CO2_Reduction_Tons'].sum()
            print(f"\n🌍 TOTAL ENVIRONMENTAL IMPACT:")
            print(f"   CO2 Reduction: {total_co2:,.0f} tons/year")
            print(f"   Equivalent cars removed: {int(total_co2/4.6):,}")
            print(f"   Equivalent trees planted: {int(total_co2*20):,}")
            
            total_savings = self.impact_results['Annual_Savings_EUR'].sum()
            total_investment = self.impact_results['Investment_EUR'].sum()
            print(f"\n💰 TOTAL ECONOMIC IMPACT:")
            print(f"   Annual Savings: EUR{total_savings:,.0f}")
            print(f"   Investment Required: EUR{total_investment:,.0f}")
            if total_savings > 0:
                print(f"   Average Payback: {total_investment/total_savings:.1f} years")
                print(f"   Average ROI: {(total_savings/total_investment)*100:.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    data_path = 'C:/Users/Zahara/Documents/Zoom/europe_energy_forecast/data/europe_energy_real.csv'
    
    analysis = CompleteEnergyAnalysis(data_path)
    
    forecast_results, impact_results = analysis.run_full_analysis()
    
    analysis.generate_report()
    
    print("\n" + "=" * 70)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - complete_energy_analysis_31_countries.csv")

if __name__ == "__main__":
    main()
