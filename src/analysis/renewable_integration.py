import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings
import re

class RenewableIntegrationAnalyzer:
    def __init__(self):
        self.hours_per_year = 8760
        self.bioenergy_keywords = ['biomass', 'biogas', 'waste', 'bioenergy', 'bioliquid']
        
    def _find_load_column(self, df: pd.DataFrame, country_code: str) -> Optional[str]:
        """Find load column for a country with flexible pattern matching."""
        country_lower = country_code.lower()
        
        # Possible column name patterns
        patterns = [
            f"{country_lower}_load_actual_entsoe_transparency",
            f"{country_lower}_load_actual",
            f"load_actual_{country_lower}",
            f"{country_lower}_load",
            f"load_{country_lower}",
            f"{country_lower}_consumption",
            f"consumption_{country_lower}"
        ]
        
        # Search for matching columns
        for col in df.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        
        # If no direct match, try regex pattern
        for col in df.columns:
            col_lower = col.lower()
            if re.search(f"{country_lower}.*load|load.*{country_lower}", col_lower):
                return col
        
        return None
    
    def _get_time_resolution(self, df: pd.DataFrame, load_series: pd.Series) -> Tuple[float, float]:
        """Determine time resolution of the dataset using median of diffs."""
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            deltas = df.index.to_series().diff().dropna()
            median_delta = deltas.median()
            delta_hours = median_delta.total_seconds() / 3600.0
            
            valid_steps = len(load_series)
            data_hours = valid_steps * delta_hours
            
            if deltas.std().total_seconds() > 1:
                warnings.warn(f"Irregular time series detected. Std dev of deltas: {deltas.std().total_seconds():.1f}s")
        else:
            delta_hours = 1.0
            data_hours = len(load_series)
            warnings.warn("No datetime index found. Assuming hourly resolution.")
        
        return delta_hours, data_hours
    
    def _identify_generation_columns(self, df: pd.DataFrame, country_prefix: str) -> Dict:
        """Identify and categorize generation columns with flexible matching."""
        country_cols = [col for col in df.columns if col.lower().startswith(country_prefix)]
        
        # Define column patterns with explicit categories
        patterns = {
            'solar': ['solar'],
            'wind': ['wind'],
            'hydro': ['hydro'],
            'nuclear': ['nuclear'],
            'bioenergy': self.bioenergy_keywords,
            'other_renewable': ['geothermal', 'tidal', 'wave'],
        }
        
        import_export_patterns = {
            'import': ['import'],
            'export': ['export']
        }
        
        identified = {key: [] for key in patterns}
        identified['import'] = []
        identified['export'] = []
        unidentified = []
        
        for col in country_cols:
            col_lower = col.lower()
            matched = False
            
            # Match generation columns
            for category, keywords in patterns.items():
                if any(keyword in col_lower for keyword in keywords):
                    if any(x in col_lower for x in ['generation', 'actual', 'production']):
                        identified[category].append(col)
                        matched = True
                        break
            
            # Match import/export columns
            if not matched:
                for category, keywords in import_export_patterns.items():
                    if any(keyword in col_lower for keyword in keywords):
                        if any(x in col_lower for x in ['actual', 'flow', 'value']):
                            if not any(x in col_lower for x in ['price', 'cost', 'tariff']):
                                identified[category].append(col)
                                matched = True
                                break
            
            if not matched and any(x in col_lower for x in ['generation', 'actual', 'production', 'load', 'consumption']):
                unidentified.append(col)
        
        if unidentified:
            print(f"     Unidentified columns: {unidentified[:5]}...")
        
        return identified
    
    def analyze_renewable_integration(self, df: pd.DataFrame, country_code: str = 'DE') -> Optional[Dict]:
        """Analyze renewable energy penetration with flexible column matching."""
        try:
            # Find load column dynamically
            load_col = self._find_load_column(df, country_code)
            
            if load_col is None:
                available_cols = [col for col in df.columns if country_code.lower() in col.lower()]
                if available_cols:
                    print(f"   Available columns for {country_code}: {available_cols[:5]}")
                raise ValueError(f"Load column for {country_code} not found in dataset")
            
            print(f"   Using load column: {load_col}")
            
            # Load data - dropna first
            load_series = df[load_col].dropna()
            if len(load_series) == 0:
                raise ValueError(f"No valid load data for {country_code}")
            
            # Get time resolution
            delta_hours, data_hours = self._get_time_resolution(df, load_series)
            print(f"\n Time resolution: {delta_hours:.3f} hours/step, {data_hours:.1f} total hours")
            print(f"   Valid load steps: {len(load_series)}/{len(df)} ({len(load_series)/len(df)*100:.1f}%)")
            
            # Correct energy calculation
            total_load_energy_mwh = load_series.sum() * delta_hours
            avg_load_mw = load_series.mean()
            
            # Identify generation columns
            country_prefix = f"{country_code.lower()}_"
            identified_cols = self._identify_generation_columns(df, country_prefix)
            
            # Calculate energy for each source
            solar_energy_mwh = 0
            if identified_cols['solar']:
                solar_data = df[identified_cols['solar']].dropna()
                if len(solar_data) > 0:
                    solar_energy_mwh = solar_data.sum().sum() * delta_hours
                    print(f"   Solar columns: {len(identified_cols['solar'])} columns")
            
            wind_energy_mwh = 0
            if identified_cols['wind']:
                wind_data = df[identified_cols['wind']].dropna()
                if len(wind_data) > 0:
                    wind_energy_mwh = wind_data.sum().sum() * delta_hours
                    print(f"   Wind columns: {len(identified_cols['wind'])} columns")
            
            hydro_energy_mwh = 0
            if identified_cols['hydro']:
                hydro_data = df[identified_cols['hydro']].dropna()
                if len(hydro_data) > 0:
                    hydro_energy_mwh = hydro_data.sum().sum() * delta_hours
                    print(f"   Hydro columns: {len(identified_cols['hydro'])} columns")
            
            nuclear_energy_mwh = 0
            if identified_cols['nuclear']:
                nuclear_data = df[identified_cols['nuclear']].dropna()
                if len(nuclear_data) > 0:
                    nuclear_energy_mwh = nuclear_data.sum().sum() * delta_hours
                    print(f"   Nuclear columns: {len(identified_cols['nuclear'])} columns")
            
            bioenergy_energy_mwh = 0
            if identified_cols['bioenergy']:
                bio_data = df[identified_cols['bioenergy']].dropna()
                if len(bio_data) > 0:
                    bioenergy_energy_mwh = bio_data.sum().sum() * delta_hours
                    print(f"   Bioenergy columns: {len(identified_cols['bioenergy'])} columns")
            
            other_renewable_energy_mwh = 0
            if identified_cols['other_renewable']:
                other_data = df[identified_cols['other_renewable']].dropna()
                if len(other_data) > 0:
                    other_renewable_energy_mwh = other_data.sum().sum() * delta_hours
                    print(f"   Other renewable columns: {len(identified_cols['other_renewable'])} columns")
            
            # Import/Export
            total_imports_mwh = 0
            if identified_cols['import']:
                import_data = df[identified_cols['import']].dropna()
                if len(import_data) > 0:
                    total_imports_mwh = import_data.sum().sum() * delta_hours
                    print(f"   Import columns: {len(identified_cols['import'])} columns")
            
            total_exports_mwh = 0
            if identified_cols['export']:
                export_data = df[identified_cols['export']].dropna()
                if len(export_data) > 0:
                    total_exports_mwh = export_data.sum().sum() * delta_hours
                    print(f"   Export columns: {len(identified_cols['export'])} columns")
            
            net_import_energy_mwh = total_imports_mwh - total_exports_mwh
            
            # Total renewable energy
            total_renewable_energy_mwh = (solar_energy_mwh + wind_energy_mwh + hydro_energy_mwh + 
                                         bioenergy_energy_mwh + other_renewable_energy_mwh)
            
            # Energy balance: Load = Domestic generation + Net imports
            domestic_generation_mwh = total_load_energy_mwh - net_import_energy_mwh
            
            # Calculate residual
            residual_energy_mwh = domestic_generation_mwh - total_renewable_energy_mwh - nuclear_energy_mwh
            
            # Check for negative residual
            if residual_energy_mwh < -0.01 * total_load_energy_mwh:
                print(f"     WARNING: Negative residual ({residual_energy_mwh:,.0f} MWh)")
                print(f"      {abs(residual_energy_mwh/total_load_energy_mwh*100):.1f}% of total load")
            
            # Calculate percentages
            solar_pct = (solar_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            wind_pct = (wind_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            hydro_pct = (hydro_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            nuclear_pct = (nuclear_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            bioenergy_pct = (bioenergy_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            other_renewable_pct = (other_renewable_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            residual_pct = (residual_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            net_import_pct = (net_import_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            
            # Average power
            total_hours = data_hours
            avg_solar_mw = solar_energy_mwh / total_hours if total_hours > 0 else 0
            avg_wind_mw = wind_energy_mwh / total_hours if total_hours > 0 else 0
            avg_hydro_mw = hydro_energy_mwh / total_hours if total_hours > 0 else 0
            avg_nuclear_mw = nuclear_energy_mwh / total_hours if total_hours > 0 else 0
            avg_bioenergy_mw = bioenergy_energy_mwh / total_hours if total_hours > 0 else 0
            avg_other_renewable_mw = other_renewable_energy_mwh / total_hours if total_hours > 0 else 0
            avg_residual_mw = residual_energy_mwh / total_hours if total_hours > 0 else 0
            avg_net_import_mw = net_import_energy_mwh / total_hours if total_hours > 0 else 0
            
            # Build result dictionary
            renewable_sources = {}
            
            if solar_energy_mwh > 0:
                renewable_sources['solar'] = {
                    'penetration_percent': round(solar_pct, 1),
                    'annual_energy_gwh': round(solar_energy_mwh / 1000, 0),
                    'avg_power_mw': round(avg_solar_mw, 0),
                    'column_count': len(identified_cols['solar'])
                }
            
            if wind_energy_mwh > 0:
                renewable_sources['wind'] = {
                    'penetration_percent': round(wind_pct, 1),
                    'annual_energy_gwh': round(wind_energy_mwh / 1000, 0),
                    'avg_power_mw': round(avg_wind_mw, 0),
                    'column_count': len(identified_cols['wind'])
                }
            
            if hydro_energy_mwh > 0:
                renewable_sources['hydro'] = {
                    'penetration_percent': round(hydro_pct, 1),
                    'annual_energy_gwh': round(hydro_energy_mwh / 1000, 0),
                    'avg_power_mw': round(avg_hydro_mw, 0),
                    'column_count': len(identified_cols['hydro'])
                }
            
            if bioenergy_energy_mwh > 0:
                renewable_sources['bioenergy'] = {
                    'penetration_percent': round(bioenergy_pct, 1),
                    'annual_energy_gwh': round(bioenergy_energy_mwh / 1000, 0),
                    'avg_power_mw': round(avg_bioenergy_mw, 0),
                    'column_count': len(identified_cols['bioenergy'])
                }
            
            if other_renewable_energy_mwh > 0:
                renewable_sources['other_renewable'] = {
                    'penetration_percent': round(other_renewable_pct, 1),
                    'annual_energy_gwh': round(other_renewable_energy_mwh / 1000, 0),
                    'avg_power_mw': round(avg_other_renewable_mw, 0),
                    'column_count': len(identified_cols['other_renewable'])
                }
            
            if nuclear_energy_mwh > 0:
                renewable_sources['nuclear'] = {
                    'penetration_percent': round(nuclear_pct, 1),
                    'annual_energy_gwh': round(nuclear_energy_mwh / 1000, 0),
                    'avg_power_mw': round(avg_nuclear_mw, 0),
                    'column_count': len(identified_cols['nuclear'])
                }
            
            # Residual
            renewable_sources['residual'] = {
                'penetration_percent': round(residual_pct, 1),
                'annual_energy_gwh': round(residual_energy_mwh / 1000, 0),
                'avg_power_mw': round(avg_residual_mw, 0),
                'note': 'Fossil fuels + storage + other non-renewable domestic generation',
                'warning': residual_energy_mwh < -0.01 * total_load_energy_mwh
            }
            
            # Net imports if significant
            if abs(net_import_pct) > 0.5:
                renewable_sources['net_imports'] = {
                    'penetration_percent': round(net_import_pct, 1),
                    'annual_energy_gwh': round(net_import_energy_mwh / 1000, 0),
                    'avg_power_mw': round(avg_net_import_mw, 0),
                    'column_count': len(identified_cols['import']) + len(identified_cols['export'])
                }
            
            return {
                'country_code': country_code,
                'load_column': load_col,
                'total_annual_load_gwh': round(total_load_energy_mwh / 1000, 0),
                'avg_load_mw': round(avg_load_mw, 0),
                'data_hours': round(total_hours, 1),
                'delta_hours': delta_hours,
                'valid_steps': len(load_series),
                'total_steps': len(df),
                'data_completeness': f"{len(load_series)/len(df)*100:.1f}%",
                'domestic_generation_gwh': round(domestic_generation_mwh / 1000, 0),
                'net_imports_gwh': round(net_import_energy_mwh / 1000, 0),
                'renewable_sources': renewable_sources,
                'methodology': {
                    'time_resolution': 'median of diffs',
                    'delta_hours': delta_hours,
                    'energy_calculation': f'Σ(MW × {delta_hours:.3f}h)',
                    'load_column_detection': 'flexible pattern matching',
                    'energy_balance': 'Load = Domestic + Net imports'
                },
                'units': {
                    'penetration': '% of annual energy consumption',
                    'energy': 'GWh/year',
                    'power': 'MW (average)',
                    'time': 'hours'
                }
            }
            
        except Exception as e:
            print(f" ERROR: Renewable integration analysis failed for {country_code}: {e}")
            return None
    
    def analyze_multiple_countries(self, df: pd.DataFrame, country_codes: Optional[List[str]] = None) -> Dict:
        """Analyze renewable integration for multiple countries."""
        if country_codes is None:
            country_codes = ['DE', 'FR', 'IT', 'ES', 'GB', 'NL', 'PL', 'BE', 'AT']
        
        results = {}
        successful = 0
        failed = 0
        
        for country_code in country_codes:
            print(f"\n{'='*50}")
            print(f"Analyzing {country_code}...")
            print(f"{'='*50}")
            
            result = self.analyze_renewable_integration(df, country_code)
            if result is not None:
                results[country_code] = result
                successful += 1
            else:
                failed += 1
        
        print(f"\n Multi-country analysis complete: {successful} successful, {failed} failed")
        return results
    
    def create_integration_report(self, df: pd.DataFrame, country_code: str = 'DE') -> Optional[Dict]:
        """Create detailed text report for a single country."""
        analysis = self.analyze_renewable_integration(df, country_code)
        
        if analysis is None:
            print(f"\n{'='*60}")
            print(f"RENEWABLE INTEGRATION ANALYSIS - {country_code}")
            print(f"{'='*60}")
            print(f"\n Cannot generate report: insufficient data")
            return None
        
        print(f"\n{'='*60}")
        print(f"RENEWABLE INTEGRATION ANALYSIS - {country_code}")
        print(f"{'='*60}")
        
        print(f"\n DATA QUALITY:")
        print(f"   Load column:        {analysis['load_column']}")
        print(f"   Valid steps:        {analysis['valid_steps']:>10,} / {analysis['total_steps']:>10,}")
        print(f"   Completeness:       {analysis['data_completeness']:>10}")
        print(f"   Time resolution:    {analysis['delta_hours']:>10.3f} hours/step (median)")
        
        print(f"\n ANNUAL SUMMARY:")
        print(f"   Total Load:         {analysis['total_annual_load_gwh']:>10,.0f} GWh")
        print(f"   Domestic Generation:{analysis['domestic_generation_gwh']:>10,.0f} GWh")
        print(f"   Net Imports:        {analysis['net_imports_gwh']:>10,.0f} GWh")
        print(f"   Average Load:       {analysis['avg_load_mw']:>10,.0f} MW")
        print(f"   Data Coverage:      {analysis['data_hours']:>10,.1f} hours")
        
        print(f"\n ENERGY MIX (% of annual consumption):")
        print(f"   {'-'*60}")
        
        renewable_sources = analysis['renewable_sources']
        total_renewable_pct = 0
        total_identified_pct = 0
        
        source_order = ['solar', 'wind', 'hydro', 'bioenergy', 'other_renewable', 
                       'nuclear', 'net_imports', 'residual']
        
        for source in source_order:
            if source in renewable_sources:
                data = renewable_sources[source]
                pct = data['penetration_percent']
                energy = data['annual_energy_gwh']
                
                if source not in ['residual', 'net_imports']:
                    if source in ['solar', 'wind', 'hydro', 'bioenergy', 'other_renewable']:
                        total_renewable_pct += pct
                    total_identified_pct += pct
                
                source_name = source.replace('_', ' ').title()
                warning_marker = '!' if data.get('warning', False) else ''
                print(f"   {source_name:18s}: {pct:5.1f}%  ({energy:>8,.0f} GWh){warning_marker}")
        
        print(f"   {'-'*60}")
        print(f"   Total Renewable:    {total_renewable_pct:5.1f}%")
        print(f"   Total Identified:   {total_identified_pct:5.1f}%")
        
        return analysis
    
    def plot_renewable_mix(self, df: pd.DataFrame, country_code: str = 'DE', 
                          save_path: Optional[str] = None, plot_type: str = 'bar'):
        """Create visualization of energy mix."""
        analysis = self.analyze_renewable_integration(df, country_code)
        
        if analysis is None:
            print(f"\n Cannot create plot for {country_code}: insufficient data")
            return None
        
        renewable_sources = analysis['renewable_sources']
        
        # Prepare data for bar chart
        sources_bar = []
        energy_bar = []
        colors_bar = []
        
        color_map = {
            'solar': '#FFD700',
            'wind': '#87CEEB',
            'hydro': '#4169E1',
            'bioenergy': '#32CD32',
            'other_renewable': '#9370DB',
            'nuclear': '#FFA500',
            'net_imports': '#A9A9A9',
            'residual': '#8B0000'
        }
        
        source_order = ['solar', 'wind', 'hydro', 'bioenergy', 'other_renewable', 
                       'nuclear', 'net_imports', 'residual']
        
        for source in source_order:
            if source in renewable_sources:
                data = renewable_sources[source]
                if data['annual_energy_gwh'] > 0:
                    source_name = source.replace('_', ' ').title()
                    if source == 'residual':
                        source_name = 'Residual (Fossil + Storage)'
                    if source == 'net_imports':
                        source_name = 'Net Imports'
                    sources_bar.append(source_name)
                    energy_bar.append(data['annual_energy_gwh'])
                    colors_bar.append(color_map.get(source, '#808080'))
        
        # Bar chart
        if plot_type in ['bar', 'both']:
            fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
            
            y_pos = np.arange(len(sources_bar))
            bars = ax_bar.barh(y_pos, energy_bar, color=colors_bar)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(sources_bar, fontsize=11)
            ax_bar.set_xlabel('Annual Energy (GWh)', fontsize=12)
            ax_bar.set_title(f'Generation by Source - {country_code}\nTotal Load: {analysis["total_annual_load_gwh"]:,.0f} GWh', 
                           fontsize=14, fontweight='bold')
            
            max_energy = max(energy_bar) if energy_bar else 0
            for i, bar in enumerate(bars):
                width = bar.get_width()
                pct = (energy_bar[i] / analysis['total_annual_load_gwh']) * 100
                label = f'{width:,.0f} GWh ({pct:.1f}%)'
                ax_bar.text(width + max_energy*0.01, bar.get_y() + bar.get_height()/2,
                          label, ha='left', va='center', fontsize=10)
            
            ax_bar.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            if save_path:
                bar_path = save_path.replace('.png', '_bar.png') if save_path.endswith('.png') else save_path
                plt.savefig(bar_path, dpi=150, bbox_inches='tight')
                print(f"   Bar chart saved to: {bar_path}")
            
            if plot_type == 'bar':
                plt.show()
        
        return fig_bar if plot_type in ['bar', 'both'] else None


def test_renewable_analyzer():
    """Test function with synthetic data."""
    print("\n" + "="*60)
    print("TESTING RENEWABLE INTEGRATION ANALYZER")
    print("="*60)
    
    hours = 876
    dates_hourly = pd.date_range('2023-01-01', periods=hours, freq='h')
    df_hourly = _create_test_dataframe(dates_hourly)
    
    analyzer = RenewableIntegrationAnalyzer()
    result = analyzer.analyze_renewable_integration(df_hourly, 'DE')
    
    if result:
        print(f"\n Test successful! Load column: {result['load_column']}")
        print(f"   Total Load: {result['total_annual_load_gwh']:.0f} GWh")
    else:
        print("\n Test failed!")
    
    return analyzer, df_hourly


def _create_test_dataframe(dates):
    """Create test dataframe with realistic column names."""
    np.random.seed(42)
    n = len(dates)
    
    data = {
        'de_load_actual': np.random.normal(50000, 5000, n),
        'de_solar_generation_actual': np.random.normal(8000, 2000, n),
        'de_wind_onshore_generation_actual': np.random.normal(12000, 3000, n),
        'de_wind_offshore_generation_actual': np.random.normal(5000, 1500, n),
        'de_hydro_generation_actual': np.random.normal(3000, 500, n),
        'de_nuclear_generation_actual': np.random.normal(8000, 200, n),
        'de_biomass_generation_actual': np.random.normal(2000, 300, n),
        'de_import_actual': np.random.normal(2000, 500, n),
        'de_export_actual': np.random.normal(1500, 400, n),
    }
    
    for col in data:
        data[col] = np.maximum(data[col], 0)
    
    df = pd.DataFrame(data, index=dates)
    return df


def main():
    print("=" * 70)
    print("RENEWABLE INTEGRATION ANALYSIS TOOL")
    print("=" * 70)
    print("\n SCIENTIFIC CORRECTIONS APPLIED:")
    print("   1. ✓ Flexible load column detection")
    print("   2. ✓ Median time resolution")
    print("   3. ✓ Valid steps only for energy calculation")
    print("   4. ✓ Stricter import/export column matching")
    print("   5. ✓ Total load threshold for negative residual")
    print("=" * 70)
    
    analyzer = RenewableIntegrationAnalyzer()
    
    try:
        # Load real data
        df = pd.read_csv('data/europe_energy_real.csv')
        print(f"\n Data loaded: {df.shape}")
        
        # Show first few columns to understand structure
        print(f"\n First 20 columns in dataset:")
        for i, col in enumerate(df.columns[:20]):
            print(f"   {i+1:2d}. {col}")
        
        # Set datetime index if timestamp column exists
        if 'utc_timestamp' in df.columns:
            df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
            df.set_index('utc_timestamp', inplace=True)
            print(f"\n   Date range: {df.index.min()} to {df.index.max()}")
        
        # Analyze key European countries
        country_codes = ['DE', 'FR', 'IT', 'ES', 'GB', 'NL', 'PL', 'BE', 'AT']
        
        print(f"\n Analyzing {len(country_codes)} countries...")
        all_results = analyzer.analyze_multiple_countries(df, country_codes)
        
        if not all_results:
            print("\n  No countries could be analyzed with real data.")
            print("   Running test with synthetic data instead...")
            analyzer, df = test_renewable_analyzer()
            analyzer.plot_renewable_mix(df, 'DE', 'test_renewable_mix.png', plot_type='bar')
        else:
            # Create summary dataframe
            print(f"\n{'='*70}")
            print("📈 EUROPEAN RENEWABLE INTEGRATION SUMMARY")
            print("=" * 70)
            
            summary_data = []
            for country_code, result in all_results.items():
                sources = result['renewable_sources']
                
                solar_pct = sources.get('solar', {}).get('penetration_percent', 0)
                wind_pct = sources.get('wind', {}).get('penetration_percent', 0)
                hydro_pct = sources.get('hydro', {}).get('penetration_percent', 0)
                bio_pct = sources.get('bioenergy', {}).get('penetration_percent', 0)
                other_re_pct = sources.get('other_renewable', {}).get('penetration_percent', 0)
                nuclear_pct = sources.get('nuclear', {}).get('penetration_percent', 0)
                residual_pct = sources.get('residual', {}).get('penetration_percent', 0)
                
                total_renewable = solar_pct + wind_pct + hydro_pct + bio_pct + other_re_pct
                
                summary_data.append({
                    'Country': country_code,
                    'Load_Column': result['load_column'],
                    'Solar_%': solar_pct,
                    'Wind_%': wind_pct,
                    'Hydro_%': hydro_pct,
                    'Total_RE_%': round(total_renewable, 1),
                    'Nuclear_%': nuclear_pct,
                    'Residual_%': residual_pct,
                    'Load_GWh': result['total_annual_load_gwh']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Total_RE_%', ascending=False)
            
            print("\n", summary_df.to_string(index=False))
            
            # Save summary
            summary_csv_path = 'renewable_integration_summary.csv'
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\n Summary saved to: {summary_csv_path}")
            
            # Create bar chart for first successful country
            first_country = list(all_results.keys())[0]
            print(f"\n Creating bar chart for {first_country}...")
            analyzer.plot_renewable_mix(df, first_country, f'{first_country}_renewable_mix.png', plot_type='bar')
        
        print(f"\n{'='*70}")
        print(" ANALYSIS COMPLETED")
        print("=" * 70)
        
    except FileNotFoundError:
        print("\n  Data file not found. Running test with synthetic data...")
        analyzer, df = test_renewable_analyzer()
        analyzer.plot_renewable_mix(df, 'DE', 'test_renewable_mix.png', plot_type='bar')
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n Running test with synthetic data...")
        analyzer, df = test_renewable_analyzer()
        analyzer.plot_renewable_mix(df, 'DE', 'test_renewable_mix.png', plot_type='bar')


if __name__ == "__main__":
    main()
