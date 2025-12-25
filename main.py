import sys
import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta

# Try to install gdown if not available
try:
    import gdown
except ImportError:
    print("Installing gdown for Google Drive download...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

# Import analysis modules
try:
    from src.analysis.carbon_impact import CarbonImpactAnalyzer
    from src.analysis.renewable_integration import RenewableIntegrationAnalyzer
    from src.analysis.economic_analysis import EconomicAnalyzer
    imports_successful = True
    print("Successfully imported analysis modules")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating analyzers for real data...")
    imports_successful = False
    
    # Updated analyzer classes for REAL DATA
    class CarbonImpactAnalyzer:
        def calculate_carbon_reduction(self, df, improvement, country_code='DE'):
            print(f"  [DEBUG] calculate_carbon_reduction for {country_code} with df shape: {df.shape}")
            try:
                # Find relevant columns for the country
                load_col = f"{country_code.lower()}_load_actual_entsoe_transparency"
                
                if load_col not in df.columns:
                    print(f"  [WARNING] Load column {load_col} not found")
                    # Try to find any load column
                    load_cols = [col for col in df.columns if 'load_actual' in col]
                    if load_cols:
                        load_col = load_cols[0]
                        print(f"  [INFO] Using alternative load column: {load_col}")
                    else:
                        return self._get_default_carbon_values()
                
                # Calculate average consumption
                avg_consumption = df[load_col].mean()
                
                # For CO2 intensity - use default European average if not available
                avg_co2 = 300  # gCO2/kWh - European average
                
                # Annual energy savings (MWh)
                # Data is hourly, so multiply by 24 for daily, then by 365
                if isinstance(df.index, pd.DatetimeIndex):
                    # Check frequency
                    time_diff = df.index[1] - df.index[0]
                    if time_diff == timedelta(hours=1):
                        hours_per_year = 8760  # 365 * 24
                    else:
                        hours_per_year = 365  # Assuming daily data
                else:
                    hours_per_year = 8760  # Default to hourly
                
                annual_energy_savings_mwh = avg_consumption * improvement * hours_per_year
                annual_co2_reduction_tons = (annual_energy_savings_mwh * avg_co2 * 1000) / 1000000
                
                equivalent_cars_removed = annual_co2_reduction_tons / 4.6
                equivalent_trees_planted = annual_co2_reduction_tons * 20
                
                result = {
                    'annual_co2_reduction_tons': float(annual_co2_reduction_tons),
                    'equivalent_cars_removed': int(equivalent_cars_removed),
                    'equivalent_trees_planted': int(equivalent_trees_planted),
                    'annual_energy_savings_mwh': float(annual_energy_savings_mwh),
                    'avg_consumption_mwh': float(avg_consumption),
                    'co2_intensity_gco2_kwh': avg_co2
                }
                
                print(f"  [DEBUG] Carbon calculation for {country_code}:")
                print(f"    Avg consumption: {avg_consumption:.0f} MWh")
                print(f"    CO2 intensity: {avg_co2} gCO2/kWh")
                print(f"    Annual energy savings: {annual_energy_savings_mwh:,.0f} MWh")
                print(f"    Annual CO2 reduction: {annual_co2_reduction_tons:,.0f} tons")
                
                return result
                
            except Exception as e:
                print(f"  [ERROR] in calculate_carbon_reduction: {e}")
                traceback.print_exc()
                return self._get_default_carbon_values()
        
        def _get_default_carbon_values(self):
            return {
                'annual_co2_reduction_tons': 50000,
                'equivalent_cars_removed': 10870,
                'equivalent_trees_planted': 1000000,
                'annual_energy_savings_mwh': 1000000,
                'avg_consumption_mwh': 50000,
                'co2_intensity_gco2_kwh': 300
            }
    
    class RenewableIntegrationAnalyzer:
        def analyze_renewable_integration(self, df, country_code='DE'):
            print(f"  [DEBUG] analyze_renewable_integration for {country_code}")
            try:
                renewable_sources = {}
                
                # Find load column
                load_col = f"{country_code.lower()}_load_actual_entsoe_transparency"
                if load_col not in df.columns:
                    load_cols = [col for col in df.columns if 'load_actual' in col]
                    if load_cols:
                        load_col = load_cols[0]
                    else:
                        return self._get_default_renewable_values()
                
                total_load = df[load_col].mean()
                if pd.isna(total_load) or total_load == 0:
                    return self._get_default_renewable_values()
                
                print(f"  [DEBUG] Total load for {country_code}: {total_load:.0f} MWh")
                
                # Find renewable generation columns for this country
                country_prefix = f"{country_code.lower()}_"
                country_cols = [col for col in df.columns if col.startswith(country_prefix)]
                
                # Categorize columns
                solar_cols = [col for col in country_cols if 'solar' in col and 'generation' in col]
                wind_cols = [col for col in country_cols if 'wind' in col and 'generation' in col]
                hydro_cols = [col for col in country_cols if 'hydro' in col and 'generation' in col]
                
                # Calculate averages
                solar_generation = 0
                if solar_cols:
                    solar_data = df[solar_cols].mean(axis=1)
                    solar_generation = solar_data.mean()
                    solar_percentage = (solar_generation / total_load) * 100
                    renewable_sources['solar'] = {
                        'penetration_percentage': round(float(solar_percentage), 1),
                        'avg_generation_mwh': round(float(solar_generation), 0)
                    }
                
                wind_generation = 0
                if wind_cols:
                    wind_data = df[wind_cols].mean(axis=1)
                    wind_generation = wind_data.mean()
                    wind_percentage = (wind_generation / total_load) * 100
                    renewable_sources['wind'] = {
                        'penetration_percentage': round(float(wind_percentage), 1),
                        'avg_generation_mwh': round(float(wind_generation), 0)
                    }
                
                hydro_generation = 0
                if hydro_cols:
                    hydro_data = df[hydro_cols].mean(axis=1)
                    hydro_generation = hydro_data.mean()
                    hydro_percentage = (hydro_generation / total_load) * 100
                    renewable_sources['hydro'] = {
                        'penetration_percentage': round(float(hydro_percentage), 1),
                        'avg_generation_mwh': round(float(hydro_generation), 0)
                    }
                
                # Calculate fossil (remainder)
                total_renewable = solar_generation + wind_generation + hydro_generation
                fossil_generation = max(0, total_load - total_renewable)
                fossil_percentage = (fossil_generation / total_load) * 100 if total_load > 0 else 0
                
                renewable_sources['fossil'] = {
                    'penetration_percentage': round(float(fossil_percentage), 1),
                    'avg_generation_mwh': round(float(fossil_generation), 0)
                }
                
                # Normalize percentages to sum to 100
                total_percentage = sum(data['penetration_percentage'] for data in renewable_sources.values())
                if total_percentage > 100:
                    scale_factor = 100 / total_percentage
                    for source in renewable_sources:
                        renewable_sources[source]['penetration_percentage'] = round(
                            renewable_sources[source]['penetration_percentage'] * scale_factor, 1
                        )
                
                result = {'renewable_sources': renewable_sources}
                
                print(f"  [DEBUG] Renewable mix for {country_code}:")
                for source, data in renewable_sources.items():
                    print(f"    {source}: {data['penetration_percentage']}% ({data['avg_generation_mwh']:,.0f} MWh)")
                
                return result
                
            except Exception as e:
                print(f"  [ERROR] in analyze_renewable_integration: {e}")
                traceback.print_exc()
                return self._get_default_renewable_values()
        
        def _get_default_renewable_values(self):
            return {
                'renewable_sources': {
                    'solar': {'penetration_percentage': 15.5, 'avg_generation_mwh': 8000},
                    'wind': {'penetration_percentage': 25.3, 'avg_generation_mwh': 12000},
                    'hydro': {'penetration_percentage': 12.7, 'avg_generation_mwh': 5000},
                    'fossil': {'penetration_percentage': 46.5, 'avg_generation_mwh': 25000}
                }
            }
    
    class EconomicAnalyzer:
        def calculate_economic_savings(self, df, improvement, co2_reduction, energy_savings_mwh=None, country_code='DE'):
            print(f"  [DEBUG] calculate_economic_savings for {country_code}")
            try:
                # Find energy price column
                price_cols = [col for col in df.columns if 'price_day_ahead' in col and country_code.lower() in col]
                
                if price_cols:
                    avg_price = df[price_cols[0]].mean()
                else:
                    # Try to find any price column
                    all_price_cols = [col for col in df.columns if 'price' in col]
                    if all_price_cols:
                        avg_price = df[all_price_cols[0]].mean()
                    else:
                        avg_price = 80  # €/MWh default European average
                
                # If energy savings not provided, calculate from improvement
                if energy_savings_mwh is None:
                    load_col = f"{country_code.lower()}_load_actual_entsoe_transparency"
                    if load_col in df.columns:
                        avg_consumption = df[load_col].mean()
                        # Data is hourly
                        energy_savings_mwh = avg_consumption * improvement * 8760  # 365 * 24
                    else:
                        energy_savings_mwh = 1000000  # default
                
                # Calculate savings
                savings_from_efficiency = energy_savings_mwh * avg_price
                carbon_price = 80  # € per ton CO2 (EU ETS average)
                savings_from_carbon = co2_reduction * carbon_price
                total_annual_savings = savings_from_efficiency + savings_from_carbon
                
                # Realistic investment calculation
                # Based on typical energy efficiency projects
                if energy_savings_mwh > 0:
                    # €500,000 per GWh annual savings
                    initial_investment = energy_savings_mwh * 500  # €500 per MWh savings
                else:
                    initial_investment = 10000000  # €10 million default
                
                # Payback period
                if total_annual_savings > 0:
                    payback_period = initial_investment / total_annual_savings
                else:
                    payback_period = 999
                
                # ROI
                roi_percentage = (total_annual_savings / initial_investment) * 100 if initial_investment > 0 else 0
                
                # NPV (20 years, 5% discount rate)
                discount_rate = 0.05
                npv = total_annual_savings * ((1 - (1 + discount_rate)**-20) / discount_rate) - initial_investment
                
                result = {
                    'total_annual_savings_eur': round(float(total_annual_savings), 0),
                    'savings_from_efficiency': round(float(savings_from_efficiency), 0),
                    'savings_from_carbon': round(float(savings_from_carbon), 0),
                    'payback_period_years': round(float(payback_period), 1),
                    'roi_percentage': round(float(roi_percentage), 1),
                    'initial_investment_eur': round(float(initial_investment), 0),
                    'npv_eur': round(float(npv), 0),
                    'energy_price_eur_per_mwh': round(float(avg_price), 1),
                    'carbon_price_eur_per_ton': carbon_price
                }
                
                print(f"  [DEBUG] Economic calculation for {country_code}:")
                print(f"    Energy price: {avg_price:.1f} €/MWh")
                print(f"    Energy savings: {energy_savings_mwh:,.0f} MWh")
                print(f"    CO2 reduction: {co2_reduction:,.0f} tons")
                print(f"    Initial investment: {initial_investment:,.0f} €")
                print(f"    Payback period: {payback_period:.1f} years")
                print(f"    ROI: {roi_percentage:.1f}%")
                
                return result
                    
            except Exception as e:
                print(f"  [ERROR] in calculate_economic_savings: {e}")
                traceback.print_exc()
                return self._get_default_economic_values()
        
        def _get_default_economic_values(self):
            return {
                'total_annual_savings_eur': 2500000,
                'savings_from_efficiency': 2000000,
                'savings_from_carbon': 500000,
                'payback_period_years': 4.0,
                'roi_percentage': 25.0,
                'initial_investment_eur': 10000000,
                'npv_eur': 15000000,
                'energy_price_eur_per_mwh': 80.0,
                'carbon_price_eur_per_ton': 80
            }
    
    print("Analyzers created successfully for real data")

def download_real_data():
    """Download real dataset from Google Drive"""
    print("\n" + "="*60)
    print("DOWNLOADING REAL DATASET")
    print("="*60)
    
    file_id = '1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s'
    output_path = 'data/europe_energy_real.csv'
    
    os.makedirs('data', exist_ok=True)
    
    if os.path.exists(output_path):
        file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(output_path))).days
        if file_age < 30:  # 30 days cache
            print(f"Using existing dataset (downloaded {file_age} days ago)")
            return output_path
    
    print("Downloading from Google Drive...")
    url = f'https://drive.google.com/uc?id={file_id}'
    
    try:
        gdown.download(url, output_path, quiet=False)
        
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024*1024)
            print(f"Download successful!")
            print(f"File: {output_path}")
            print(f"Size: {size_mb:.2f} MB")
            return output_path
            
    except Exception as e:
        print(f"Download error: {e}")
    
    return None

def load_and_prepare_data(use_real_data=True, sample_size=None):
    """Load and prepare data for analysis"""
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    if use_real_data:
        data_path = download_real_data()
        
        if data_path and os.path.exists(data_path):
            try:
                print(f"\nReading real dataset...")
                
                # Read only first N rows if sample_size specified (for testing)
                if sample_size:
                    df = pd.read_csv(data_path, nrows=sample_size)
                    print(f"Loaded sample of {sample_size} rows")
                else:
                    df = pd.read_csv(data_path)
                
                print(f"Real dataset loaded successfully!")
                print(f"Shape: {df.shape}")
                print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
                
                # Standardize column names
                df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
                
                # Handle datetime index
                time_cols = [col for col in df.columns if 'timestamp' in col or 'date' in col or 'time' in col]
                if time_cols:
                    time_col = time_cols[0]
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    df.set_index(time_col, inplace=True)
                    print(f"Set '{time_col}' as index")
                    print(f"Time range: {df.index.min()} to {df.index.max()}")
                    
                    # Sort by time
                    df = df.sort_index()
                
                # Fill missing values with forward fill for time series
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
                
                # Remove completely empty columns
                df = df.dropna(axis=1, how='all')
                
                print(f"\nFinal dataset shape: {df.shape}")
                print(f"Available countries (based on column prefixes):")
                country_codes = set()
                for col in df.columns:
                    if '_' in col:
                        prefix = col.split('_')[0]
                        if len(prefix) == 2 and prefix.isalpha():
                            country_codes.add(prefix.upper())
                
                if country_codes:
                    print(f"  {', '.join(sorted(country_codes))}")
                
                return df
                
            except Exception as e:
                print(f"Error loading real data: {e}")
                print("Falling back to sample data...")
                return load_sample_data()
        else:
            print("Real dataset not available. Using sample data...")
            return load_sample_data()
    else:
        print("Using sample data for demonstration...")
        return load_sample_data()

def load_sample_data():
    """Create sample energy data for demonstration"""
    print("[DEBUG] Creating sample data...")
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    
    base_load = 50000
    seasonal_variation = 10000 * np.sin(2 * np.pi * np.arange(365) / 365)
    
    data = {
        'date': dates,
        'de_load_actual_entsoe_transparency': base_load + seasonal_variation + np.random.normal(0, 5000, 365),
        'de_solar_generation_actual': 8000 + 3000 * np.sin(2 * np.pi * np.arange(365) / 365 + np.pi/2),
        'de_wind_generation_actual': 12000 + np.random.normal(0, 4000, 365),
        'de_wind_onshore_generation_actual': 10000 + np.random.normal(0, 3000, 365),
        'de_wind_offshore_generation_actual': 2000 + np.random.normal(0, 1000, 365),
        'de_price_day_ahead': 80 + 20 * np.sin(2 * np.pi * np.arange(365) / 365 * 2) + np.random.normal(0, 5, 365)
    }
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    for col in df.columns:
        df[col] = df[col].clip(lower=1)
    
    print(f"[DEBUG] Sample data created with shape: {df.shape}")
    return df

def analyze_data_structure(df):
    """Analyze the structure of the loaded data"""
    print("\n" + "="*60)
    print("DATA STRUCTURE ANALYSIS")
    print("="*60)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    
    # Detect country codes
    country_codes = set()
    for col in df.columns:
        if '_' in col:
            prefix = col.split('_')[0]
            if len(prefix) == 2 and prefix.isalpha():
                country_codes.add(prefix.upper())
    
    if country_codes:
        print(f"\nDetected country codes: {', '.join(sorted(country_codes))}")
        
        # Show column categories for first 3 countries
        for country in sorted(country_codes)[:3]:
            country_cols = [col for col in df.columns if col.startswith(country.lower() + '_')]
            print(f"\n{country} columns ({len(country_cols)}):")
            
            # Categorize
            load_cols = [col for col in country_cols if 'load' in col]
            solar_cols = [col for col in country_cols if 'solar' in col]
            wind_cols = [col for col in country_cols if 'wind' in col]
            price_cols = [col for col in country_cols if 'price' in col]
            
            if load_cols:
                print(f"  Load: {len(load_cols)} columns")
            if solar_cols:
                print(f"  Solar: {len(solar_cols)} columns")
            if wind_cols:
                print(f"  Wind: {len(wind_cols)} columns")
            if price_cols:
                print(f"  Price: {len(price_cols)} columns")
    
    return df

def select_country_for_analysis(df):
    """Let user select a country for analysis"""
    country_codes = set()
    for col in df.columns:
        if '_' in col:
            prefix = col.split('_')[0]
            if len(prefix) == 2 and prefix.isalpha():
                country_codes.add(prefix.upper())
    
    if not country_codes:
        return 'DE'  # Default to Germany
    
    print(f"\nAvailable countries: {', '.join(sorted(country_codes))}")
    
    # Check which countries have sufficient data
    valid_countries = []
    for country in sorted(country_codes):
        load_col = f"{country.lower()}_load_actual_entsoe_transparency"
        if load_col in df.columns:
            valid_countries.append(country)
    
    if not valid_countries:
        return 'DE'
    
    print(f"Countries with load data: {', '.join(valid_countries)}")
    
    # Prefer Germany if available (most complete data)
    if 'DE' in valid_countries:
        print("Selecting Germany (DE) - most complete data")
        return 'DE'
    
    # Otherwise use first available
    print(f"Selecting {valid_countries[0]} - first available with load data")
    return valid_countries[0]

def main():
    print("=" * 80)
    print("EUROPE ENERGY FORECAST - REAL DATA ANALYSIS")
    print("=" * 80)
    
    try:
        # Configuration
        use_real_data = True
        improvement = 0.15  # 15% improvement
        
        # 1. Load data
        print(f"\n1. Loading {'REAL' if use_real_data else 'SAMPLE'} data...")
        
        # For testing, use smaller sample of real data
        df = load_and_prepare_data(use_real_data=use_real_data, sample_size=10000)
        
        analyze_data_structure(df)
        
        # 2. Select country for analysis
        target_country = select_country_for_analysis(df)
        print(f"\n2. Target country: {target_country}")
        print(f"   Improvement factor: {improvement:.1%}")
        
        # 3. Initialize analyzers
        print("\n" + "=" * 80)
        print("3. ENVIRONMENTAL & ECONOMIC IMPACT ANALYSIS")
        print("=" * 80)
        
        print("\nInitializing analyzers...")
        if imports_successful:
            print("Using real analyzers from src.analysis")
            carbon_analyzer = CarbonImpactAnalyzer()
            renewable_analyzer = RenewableIntegrationAnalyzer()
            economic_analyzer = EconomicAnalyzer()
        else:
            print("Using built-in analyzers for real data")
            carbon_analyzer = CarbonImpactAnalyzer()
            renewable_analyzer = RenewableIntegrationAnalyzer()
            economic_analyzer = EconomicAnalyzer()
        
        # 4. Perform analyses
        print(f"\nA. Calculating carbon impact for {target_country}...")
        carbon_impact = carbon_analyzer.calculate_carbon_reduction(df, improvement, target_country)
        print(f"   ✓ Carbon impact calculated")
        
        print(f"\nB. Analyzing renewable integration for {target_country}...")
        renewable_analysis = renewable_analyzer.analyze_renewable_integration(df, target_country)
        print(f"   ✓ Renewable integration analyzed")
        
        print(f"\nC. Calculating economic impact for {target_country}...")
        co2_reduction = carbon_impact.get('annual_co2_reduction_tons', 0)
        energy_savings = carbon_impact.get('annual_energy_savings_mwh', 0)
        
        print(f"   Using:")
        print(f"     CO2 reduction: {co2_reduction:,.0f} tons")
        print(f"     Energy savings: {energy_savings:,.0f} MWh")
        
        economic_impact = economic_analyzer.calculate_economic_savings(
            df, improvement, co2_reduction, energy_savings, target_country
        )
        print(f"   ✓ Economic impact calculated")
        
        # 5. Display results
        print("\n" + "=" * 80)
        print("4. COMPREHENSIVE RESULTS")
        print("=" * 80)
        
        print(f"\nRESULTS FOR {target_country}")
        print("-" * 40)
        
        if carbon_impact:
            print(f"\nCARBON REDUCTION IMPACT:")
            print(f"  • Average consumption: {carbon_impact.get('avg_consumption_mwh', 0):,.0f} MWh")
            print(f"  • CO2 intensity: {carbon_impact.get('co2_intensity_gco2_kwh', 0)} gCO2/kWh")
            print(f"  • Annual energy savings: {carbon_impact.get('annual_energy_savings_mwh', 0):,.0f} MWh")
            print(f"  • Annual CO2 reduction: {carbon_impact.get('annual_co2_reduction_tons', 0):,.0f} tons")
            print(f"  • Equivalent to removing {carbon_impact.get('equivalent_cars_removed', 0):,.0f} cars")
            print(f"  • Or planting {carbon_impact.get('equivalent_trees_planted', 0):,.0f} trees")
        
        if economic_impact:
            print(f"\nECONOMIC IMPACT ANALYSIS:")
            print(f"  • Energy price: {economic_impact.get('energy_price_eur_per_mwh', 0):.1f} €/MWh")
            print(f"  • Carbon price: {economic_impact.get('carbon_price_eur_per_ton', 0)} €/ton")
            print(f"  • Initial investment: {economic_impact.get('initial_investment_eur', 0):,.0f} €")
            print(f"  • Annual savings breakdown:")
            print(f"     - Energy efficiency: {economic_impact.get('savings_from_efficiency', 0):,.0f} €")
            print(f"     - Carbon credits: {economic_impact.get('savings_from_carbon', 0):,.0f} €")
            print(f"  • Total annual savings: {economic_impact.get('total_annual_savings_eur', 0):,.0f} €")
            print(f"  • Payback period: {economic_impact.get('payback_period_years', 0):.1f} years")
            print(f"  • ROI: {economic_impact.get('roi_percentage', 0):.1f}%")
            print(f"  • Net Present Value (20 years): {economic_impact.get('npv_eur', 0):,.0f} €")
        
        if renewable_analysis and 'renewable_sources' in renewable_analysis:
            print(f"\nCURRENT ENERGY MIX:")
            total_renewable = 0
            
            for source in ['solar', 'wind', 'hydro', 'fossil']:
                if source in renewable_analysis['renewable_sources']:
                    data = renewable_analysis['renewable_sources'][source]
                    percentage = data.get('penetration_percentage', 0)
                    generation = data.get('avg_generation_mwh', 0)
                    
                    if source != 'fossil':
                        total_renewable += percentage
                    
                    source_display = {
                        'solar': 'Solar',
                        'wind': 'Wind',
                        'hydro': 'Hydro',
                        'fossil': 'Fossil'
                    }[source]
                    
                    print(f"  • {source_display}: {percentage:.1f}% ({generation:,.0f} MWh)")
            
            print(f"\n  SUMMARY:")
            print(f"    Total renewable: {total_renewable:.1f}%")
            print(f"    Fossil fuels: {renewable_analysis['renewable_sources'].get('fossil', {}).get('penetration_percentage', 0):.1f}%")
        
        # 6. Recommendations
        print("\n" + "=" * 80)
        print("5. RECOMMENDATIONS")
        print("=" * 80)
        
        if renewable_analysis and 'renewable_sources' in renewable_analysis:
            fossil_percentage = renewable_analysis['renewable_sources'].get('fossil', {}).get('penetration_percentage', 0)
            
            print(f"\nENERGY TRANSITION STRATEGY:")
            if fossil_percentage > 60:
                print(f"  ⚠️  CRITICAL: Very high fossil dependency ({fossil_percentage:.1f}%)")
                print(f"  ✅ Priority actions:")
                print(f"    1. Accelerate wind and solar deployment")
                print(f"    2. Invest in grid infrastructure and energy storage")
                print(f"    3. Implement energy efficiency programs")
            elif fossil_percentage > 40:
                print(f"  ⚠️  HIGH: Significant fossil dependency ({fossil_percentage:.1f}%)")
                print(f"  ✅ Recommended actions:")
                print(f"    1. Expand renewable capacity")
                print(f"    2. Modernize grid with smart technologies")
                print(f"    3. Promote electric vehicles and heat pumps")
            else:
                print(f"  ✅ GOOD: Moderate fossil dependency ({fossil_percentage:.1f}%)")
                print(f"  ✅ Continue with:")
                print(f"    1. Grid flexibility enhancement")
                print(f"    2. Demand response programs")
                print(f"    3. Sector coupling (power-to-X)")
        
        if economic_impact:
            payback = economic_impact.get('payback_period_years', 0)
            roi = economic_impact.get('roi_percentage', 0)
            
            print(f"\nINVESTMENT ASSESSMENT:")
            if payback < 3 and roi > 20:
                print(f"  ✅ EXCELLENT: Very attractive investment")
                print(f"     Payback: {payback:.1f} years, ROI: {roi:.1f}%")
                print(f"  🎯 Recommendation: Immediate implementation")
            elif payback < 7 and roi > 10:
                print(f"  ✅ GOOD: Attractive investment")
                print(f"     Payback: {payback:.1f} years, ROI: {roi:.1f}%")
                print(f"  🎯 Recommendation: Phased implementation")
            else:
                print(f"  ⚠️  CHALLENGING: Requires careful consideration")
                print(f"     Payback: {payback:.1f} years, ROI: {roi:.1f}%")
                print(f"  🎯 Recommendation:")
                print(f"     - Seek government incentives/subsidies")
                print(f"     - Consider technology alternatives")
                print(f"     - Pilot project before full scale")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Save results to file
        save_results_to_file(target_country, carbon_impact, renewable_analysis, economic_impact)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("ANALYSIS FAILED")
        print("=" * 80)
        return 1
    
    return 0

def save_results_to_file(country, carbon_impact, renewable_analysis, economic_impact):
    """Save analysis results to a CSV file"""
    try:
        os.makedirs('outputs', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/analysis_results_{country}_{timestamp}.csv"
        
        results = {
            'Country': [country],
            'Timestamp': [timestamp],
            'Annual_CO2_Reduction_tons': [carbon_impact.get('annual_co2_reduction_tons', 0)],
            'Annual_Energy_Savings_MWh': [carbon_impact.get('annual_energy_savings_mwh', 0)],
            'Equivalent_Cars_Removed': [carbon_impact.get('equivalent_cars_removed', 0)],
            'Solar_Percentage': [renewable_analysis.get('renewable_sources', {}).get('solar', {}).get('penetration_percentage', 0)],
            'Wind_Percentage': [renewable_analysis.get('renewable_sources', {}).get('wind', {}).get('penetration_percentage', 0)],
            'Hydro_Percentage': [renewable_analysis.get('renewable_sources', {}).get('hydro', {}).get('penetration_percentage', 0)],
            'Fossil_Percentage': [renewable_analysis.get('renewable_sources', {}).get('fossil', {}).get('penetration_percentage', 0)],
            'Total_Annual_Savings_EUR': [economic_impact.get('total_annual_savings_eur', 0)],
            'Payback_Period_Years': [economic_impact.get('payback_period_years', 0)],
            'ROI_Percentage': [economic_impact.get('roi_percentage', 0)],
            'Initial_Investment_EUR': [economic_impact.get('initial_investment_eur', 0)],
            'NPV_EUR': [economic_impact.get('npv_eur', 0)]
        }
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
    except Exception as e:
        print(f"Warning: Could not save results to file: {e}")

if __name__ == "__main__":
    # Install required packages
    try:
        import gdown
    except ImportError:
        print("Installing gdown package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    
    exit_code = main()
    sys.exit(exit_code)
