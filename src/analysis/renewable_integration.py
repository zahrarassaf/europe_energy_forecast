import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings
import re
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class RenewableIntegrationAnalyzer:
    def __init__(self):
        self.hours_per_year = 8760
        self.bioenergy_keywords = ['biomass', 'biogas', 'waste', 'bioenergy', 'bioliquid']
        
        self.default_emission_factors = {
            'fossil_coal': 820,
            'fossil_gas': 490,
            'fossil_oil': 650,
            'fossil_peat': 380,
            'fossil_other': 600,
            'solar': 40,
            'wind': 11,
            'hydro': 24,
            'nuclear': 12,
            'biomass': 230,
            'geothermal': 38
        }
        
        self.eu_targets = {
            '2030_renewable_pct': 42.5,
            '2030_emission_reduction_pct': 55,
            '2050_renewable_pct': 100,
            '2050_emission_reduction_pct': 100
        }
    
    def _find_load_column(self, df: pd.DataFrame, country_code: str) -> Optional[str]:
        country_upper = country_code.upper()
        country_lower = country_code.lower()
        
        patterns = [
            f"{country_upper}_load_actual_entsoe_transparency",
            f"{country_lower}_load_actual_entsoe_transparency",
            f"{country_upper}_load_actual",
            f"{country_lower}_load_actual",
            f"load_actual_{country_lower}",
            f"{country_upper}_load",
            f"{country_lower}_load",
            f"load_{country_lower}",
            f"{country_upper}_consumption",
            f"{country_lower}_consumption",
        ]
        
        for col in df.columns:
            col_upper = col.upper()
            for pattern in patterns:
                if pattern.upper() in col_upper:
                    return col
        
        for col in df.columns:
            col_upper = col.upper()
            if re.search(f"{country_upper}.*LOAD|LOAD.*{country_upper}", col_upper):
                return col
        
        return None
    
    def _get_time_resolution(self, df: pd.DataFrame, load_series: pd.Series) -> Tuple[float, float]:
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            deltas = df.index.to_series().diff().dropna()
            median_delta = deltas.median()
            delta_hours = median_delta.total_seconds() / 3600.0
            valid_steps = len(load_series)
            data_hours = valid_steps * delta_hours
        else:
            delta_hours = 1.0
            data_hours = len(load_series)
        
        return delta_hours, data_hours
    
    def _identify_all_columns_for_country(self, df: pd.DataFrame, country_code: str) -> Dict:
        country_upper = country_code.upper()
        
        results = {
            'load': [],
            'solar': [],
            'wind': [],
            'hydro': [],
            'nuclear': [],
            'biomass': [],
            'geothermal': [],
            'fossil_coal': [],
            'fossil_gas': [],
            'fossil_oil': [],
            'fossil_peat': [],
            'fossil_other': [],
            'import': [],
            'export': []
        }
        
        for col in df.columns:
            col_upper = col.upper()
            
            if not col_upper.startswith(country_upper):
                if f"_{country_upper}_" not in col_upper and not col_upper.endswith(f"_{country_upper}"):
                    continue
            
            col_lower = col.lower()
            
            if 'load_actual' in col_lower and 'forecast' not in col_lower:
                results['load'].append(col)
            elif 'solar' in col_lower and 'generation' in col_lower:
                results['solar'].append(col)
            elif 'wind' in col_lower and 'generation' in col_lower:
                results['wind'].append(col)
            elif 'hydro' in col_lower and 'generation' in col_lower:
                results['hydro'].append(col)
            elif 'nuclear' in col_lower and 'generation' in col_lower:
                results['nuclear'].append(col)
            elif any(bio in col_lower for bio in ['biomass', 'biogas', 'waste']) and 'generation' in col_lower:
                results['biomass'].append(col)
            elif 'geothermal' in col_lower and 'generation' in col_lower:
                results['geothermal'].append(col)
            elif any(coal in col_lower for coal in ['coal', 'lignite']) and 'generation' in col_lower:
                results['fossil_coal'].append(col)
            elif 'gas' in col_lower and 'generation' in col_lower and 'biogas' not in col_lower:
                results['fossil_gas'].append(col)
            elif 'oil' in col_lower and 'generation' in col_lower:
                results['fossil_oil'].append(col)
            elif 'peat' in col_lower and 'generation' in col_lower:
                results['fossil_peat'].append(col)
            elif 'import' in col_lower and 'actual' in col_lower and 'forecast' not in col_lower:
                results['import'].append(col)
            elif 'export' in col_lower and 'actual' in col_lower and 'forecast' not in col_lower:
                results['export'].append(col)
        
        return results
    
    def load_and_split_tab_file(self, file_path: str) -> pd.DataFrame:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            if ';' in first_line and '\t' in first_line:
                df = pd.read_csv(file_path, sep='\t')
                if len(df.columns) == 1:
                    col_names = df.columns[0].split(';')
                    df = df[df.columns[0]].str.split(';', expand=True)
                    df.columns = col_names
                    return df
                return df
            elif '\t' in first_line:
                df = pd.read_csv(file_path, sep='\t')
                if len(df.columns) == 1 and ';' in df.columns[0]:
                    col_names = df.columns[0].split(';')
                    df = df[df.columns[0]].str.split(';', expand=True)
                    df.columns = col_names
                    return df
                return df
            elif ';' in first_line:
                df = pd.read_csv(file_path, sep=';')
                if len(df.columns) == 1 and '\t' in df.columns[0]:
                    col_names = df.columns[0].split('\t')
                    df = df[df.columns[0]].str.split('\t', expand=True)
                    df.columns = col_names
                    return df
                return df
            else:
                return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def analyze_from_2024_data(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        print("\n" + "="*70)
        print("ANALYZING ENTSO-E 2024 DATA")
        print("="*70)
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(base_dir, 'data')
        
        load_file = os.path.join(data_dir, 'monthly_hourly_load_values_2024.csv')
        gen_file = os.path.join(data_dir, 'monthly_domestic_values_2024.csv')
        flows_file = os.path.join(data_dir, 'physical_energy_power_flows_2024.csv')
        
        print(f"Looking for data in: {data_dir}")
        
        if not os.path.exists(load_file):
            print(f"   File not found: {load_file}")
            return None
        
        if not os.path.exists(gen_file):
            print(f"   File not found: {gen_file}")
            return None
        
        if not os.path.exists(flows_file):
            print(f"   File not found: {flows_file}")
            return None
        
        try:
            df_load = pd.read_csv(load_file, sep='\t')
            
            df_gen = self.load_and_split_tab_file(gen_file)
            if df_gen is None:
                return None
            
            df_flows = self.load_and_split_tab_file(flows_file)
            if df_flows is None:
                return None
            
        except Exception as e:
            print(f"Error loading files: {e}")
            return None
        
        print(f"   Load data: {df_load.shape[0]:,} rows")
        print(f"   Generation data: {df_gen.shape[0]:,} rows")
        print(f"   Flows data: {df_flows.shape[0]:,} rows")
        print(f"   Gen columns: {df_gen.columns.tolist()}")
        print(f"   Flows columns: {df_flows.columns.tolist()}")
        
        area_col = 'Area'
        category_col = 'Category'
        value_col = 'ProvidedValue'
        
        if area_col not in df_gen.columns:
            for col in df_gen.columns:
                if 'area' in col.lower():
                    area_col = col
                    break
        
        if category_col not in df_gen.columns:
            for col in df_gen.columns:
                if 'category' in col.lower():
                    category_col = col
                    break
        
        if value_col not in df_gen.columns:
            for col in df_gen.columns:
                if 'providedvalue' in col.lower() and 'code' not in col.lower():
                    value_col = col
                    break
        
        print(f"   Using generation columns: Area='{area_col}', Category='{category_col}', Value='{value_col}'")
        
        df_gen[value_col] = pd.to_numeric(df_gen[value_col], errors='coerce')
        print(f"   Total generation sum: {df_gen[value_col].sum():,.0f} GWh")
        
        from_area_col = 'FromAreaCode'
        to_area_col = 'ToAreaCode'
        direction_col = 'Direction'
        value_flow_col = 'ProvidedValue'
        
        if from_area_col not in df_flows.columns:
            for col in df_flows.columns:
                if 'fromareacode' in col.lower():
                    from_area_col = col
                    break
        
        if to_area_col not in df_flows.columns:
            for col in df_flows.columns:
                if 'toareacode' in col.lower():
                    to_area_col = col
                    break
        
        if direction_col not in df_flows.columns:
            for col in df_flows.columns:
                if 'direction' in col.lower():
                    direction_col = col
                    break
        
        if value_flow_col not in df_flows.columns:
            for col in df_flows.columns:
                if 'providedvalue' in col.lower():
                    value_flow_col = col
                    break
        
        print(f"   Using flow columns: FromArea='{from_area_col}', ToArea='{to_area_col}', Direction='{direction_col}', Value='{value_flow_col}'")
        
        df_flows[value_flow_col] = pd.to_numeric(df_flows[value_flow_col], errors='coerce')
        
        countries = df_load['CountryCode'].unique()
        results = []
        
        for country in countries:
            country_load = df_load[df_load['CountryCode'] == country]
            if len(country_load) == 0:
                continue
            total_load_gwh = country_load['Value'].sum() / 1000
            
            country_gen = df_gen[df_gen[area_col].str.strip() == country]
            if len(country_gen) == 0:
                continue
            
            renewable_cats = ['Wind Onshore', 'Wind Offshore', 'Solar', 'Hydro Run-of-river and poundage', 
                              'Hydro Water Reservoir', 'Biomass', 'Geothermal', 'Other renewable']
            renewable_gen = country_gen[country_gen[category_col].isin(renewable_cats)][value_col].sum()
            
            nuclear_gen = country_gen[country_gen[category_col] == 'Nuclear'][value_col].sum()
            
            fossil_cats = ['Fossil Gas', 'Fossil Hard coal', 'Fossil Brown coal/Lignite', 
                           'Fossil Oil', 'Fossil Peat', 'Fossil Coal-derived gas']
            fossil_gen = country_gen[country_gen[category_col].isin(fossil_cats)][value_col].sum()
            
            imports = df_flows[(df_flows[to_area_col] == country) & (df_flows[direction_col] == 'Import')][value_flow_col].sum()
            exports = df_flows[(df_flows[from_area_col] == country) & (df_flows[direction_col] == 'Export')][value_flow_col].sum()
            net_imports = imports - exports
            
            if total_load_gwh > 0:
                results.append({
                    'Country': country,
                    'Load_GWh': round(total_load_gwh),
                    'Renewable_GWh': round(renewable_gen),
                    'Renewable_Pct': round(renewable_gen / total_load_gwh * 100, 1),
                    'Nuclear_GWh': round(nuclear_gen),
                    'Nuclear_Pct': round(nuclear_gen / total_load_gwh * 100, 1),
                    'Fossil_GWh': round(fossil_gen),
                    'Fossil_Pct': round(fossil_gen / total_load_gwh * 100, 1),
                    'Net_Imports_GWh': round(net_imports),
                    'Net_Imports_Pct': round(net_imports / total_load_gwh * 100, 1)
                })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Renewable_Pct', ascending=False)
        
        return results_df, df_gen, df_load, df_flows
    
    def analyze_renewable_integration(self, df: pd.DataFrame, country_code: str = 'DE') -> Optional[Dict]:
        try:
            print(f"\n Looking for data for country: {country_code}")
            
            identified_cols = self._identify_all_columns_for_country(df, country_code)
            
            if not identified_cols['load']:
                print(f"   No load column found for {country_code}")
                return None
            
            load_col = identified_cols['load'][0]
            print(f"   Using load column: {load_col}")
            
            load_series = df[load_col].dropna()
            if len(load_series) == 0:
                print(f"   No valid load data for {country_code}")
                return None
            
            delta_hours, data_hours = self._get_time_resolution(df, load_series)
            print(f"   Time resolution: {delta_hours:.3f} hours/step")
            print(f"   Valid load steps: {len(load_series)}/{len(df)} ({len(load_series)/len(df)*100:.1f}%)")
            
            total_load_energy_mwh = load_series.sum() * delta_hours
            avg_load_mw = load_series.mean()
            
            solar_energy_mwh = 0
            if identified_cols['solar']:
                solar_data = df[identified_cols['solar']].dropna()
                if len(solar_data) > 0:
                    solar_energy_mwh = solar_data.sum().sum() * delta_hours
            
            wind_energy_mwh = 0
            if identified_cols['wind']:
                wind_data = df[identified_cols['wind']].dropna()
                if len(wind_data) > 0:
                    wind_energy_mwh = wind_data.sum().sum() * delta_hours
            
            hydro_energy_mwh = 0
            if identified_cols['hydro']:
                hydro_data = df[identified_cols['hydro']].dropna()
                if len(hydro_data) > 0:
                    hydro_energy_mwh = hydro_data.sum().sum() * delta_hours
            
            nuclear_energy_mwh = 0
            if identified_cols['nuclear']:
                nuclear_data = df[identified_cols['nuclear']].dropna()
                if len(nuclear_data) > 0:
                    nuclear_energy_mwh = nuclear_data.sum().sum() * delta_hours
            
            biomass_energy_mwh = 0
            if identified_cols['biomass']:
                biomass_data = df[identified_cols['biomass']].dropna()
                if len(biomass_data) > 0:
                    biomass_energy_mwh = biomass_data.sum().sum() * delta_hours
            
            geothermal_energy_mwh = 0
            if identified_cols['geothermal']:
                geo_data = df[identified_cols['geothermal']].dropna()
                if len(geo_data) > 0:
                    geothermal_energy_mwh = geo_data.sum().sum() * delta_hours
            
            fossil_coal_mwh = 0
            if identified_cols['fossil_coal']:
                coal_data = df[identified_cols['fossil_coal']].dropna()
                if len(coal_data) > 0:
                    fossil_coal_mwh = coal_data.sum().sum() * delta_hours
            
            fossil_gas_mwh = 0
            if identified_cols['fossil_gas']:
                gas_data = df[identified_cols['fossil_gas']].dropna()
                if len(gas_data) > 0:
                    fossil_gas_mwh = gas_data.sum().sum() * delta_hours
            
            fossil_oil_mwh = 0
            if identified_cols['fossil_oil']:
                oil_data = df[identified_cols['fossil_oil']].dropna()
                if len(oil_data) > 0:
                    fossil_oil_mwh = oil_data.sum().sum() * delta_hours
            
            fossil_peat_mwh = 0
            if identified_cols['fossil_peat']:
                peat_data = df[identified_cols['fossil_peat']].dropna()
                if len(peat_data) > 0:
                    fossil_peat_mwh = peat_data.sum().sum() * delta_hours
            
            total_imports_mwh = 0
            if identified_cols['import']:
                import_data = df[identified_cols['import']].dropna()
                if len(import_data) > 0:
                    total_imports_mwh = import_data.sum().sum() * delta_hours
            
            total_exports_mwh = 0
            if identified_cols['export']:
                export_data = df[identified_cols['export']].dropna()
                if len(export_data) > 0:
                    total_exports_mwh = export_data.sum().sum() * delta_hours
            
            net_import_energy_mwh = total_imports_mwh - total_exports_mwh
            
            total_renewable_energy_mwh = (solar_energy_mwh + wind_energy_mwh + hydro_energy_mwh + 
                                         biomass_energy_mwh + geothermal_energy_mwh)
            
            total_fossil_energy_mwh = (fossil_coal_mwh + fossil_gas_mwh + fossil_oil_mwh + fossil_peat_mwh)
            
            domestic_generation_mwh = total_load_energy_mwh - net_import_energy_mwh
            calculated_generation = total_renewable_energy_mwh + nuclear_energy_mwh + total_fossil_energy_mwh
            residual_energy_mwh = domestic_generation_mwh - calculated_generation
            
            solar_pct = (solar_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            wind_pct = (wind_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            hydro_pct = (hydro_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            nuclear_pct = (nuclear_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            biomass_pct = (biomass_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            geothermal_pct = (geothermal_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            fossil_coal_pct = (fossil_coal_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            fossil_gas_pct = (fossil_gas_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            fossil_oil_pct = (fossil_oil_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            fossil_peat_pct = (fossil_peat_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            residual_pct = (residual_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            net_import_pct = (net_import_energy_mwh / total_load_energy_mwh) * 100 if total_load_energy_mwh > 0 else 0
            
            renewable_sources = {}
            
            if solar_energy_mwh > 0:
                renewable_sources['solar'] = {
                    'penetration_percent': round(solar_pct, 1),
                    'annual_energy_gwh': round(solar_energy_mwh / 1000, 0),
                    'avg_power_mw': round(solar_energy_mwh / data_hours, 0) if data_hours > 0 else 0
                }
            
            if wind_energy_mwh > 0:
                renewable_sources['wind'] = {
                    'penetration_percent': round(wind_pct, 1),
                    'annual_energy_gwh': round(wind_energy_mwh / 1000, 0),
                    'avg_power_mw': round(wind_energy_mwh / data_hours, 0) if data_hours > 0 else 0
                }
            
            if hydro_energy_mwh > 0:
                renewable_sources['hydro'] = {
                    'penetration_percent': round(hydro_pct, 1),
                    'annual_energy_gwh': round(hydro_energy_mwh / 1000, 0),
                    'avg_power_mw': round(hydro_energy_mwh / data_hours, 0) if data_hours > 0 else 0
                }
            
            if biomass_energy_mwh > 0:
                renewable_sources['biomass'] = {
                    'penetration_percent': round(biomass_pct, 1),
                    'annual_energy_gwh': round(biomass_energy_mwh / 1000, 0),
                    'avg_power_mw': round(biomass_energy_mwh / data_hours, 0) if data_hours > 0 else 0
                }
            
            if geothermal_energy_mwh > 0:
                renewable_sources['geothermal'] = {
                    'penetration_percent': round(geothermal_pct, 1),
                    'annual_energy_gwh': round(geothermal_energy_mwh / 1000, 0),
                    'avg_power_mw': round(geothermal_energy_mwh / data_hours, 0) if data_hours > 0 else 0
                }
            
            if nuclear_energy_mwh > 0:
                renewable_sources['nuclear'] = {
                    'penetration_percent': round(nuclear_pct, 1),
                    'annual_energy_gwh': round(nuclear_energy_mwh / 1000, 0),
                    'avg_power_mw': round(nuclear_energy_mwh / data_hours, 0) if data_hours > 0 else 0
                }
            
            if fossil_coal_mwh > 0:
                renewable_sources['fossil_coal'] = {
                    'penetration_percent': round(fossil_coal_pct, 1),
                    'annual_energy_gwh': round(fossil_coal_mwh / 1000, 0),
                    'avg_power_mw': round(fossil_coal_mwh / data_hours, 0) if data_hours > 0 else 0
                }
            
            if fossil_gas_mwh > 0:
                renewable_sources['fossil_gas'] = {
                    'penetration_percent': round(fossil_gas_pct, 1),
                    'annual_energy_gwh': round(fossil_gas_mwh / 1000, 0),
                    'avg_power_mw': round(fossil_gas_mwh / data_hours, 0) if data_hours > 0 else 0
                }
            
            if fossil_oil_mwh > 0:
                renewable_sources['fossil_oil'] = {
                    'penetration_percent': round(fossil_oil_pct, 1),
                    'annual_energy_gwh': round(fossil_oil_mwh / 1000, 0),
                    'avg_power_mw': round(fossil_oil_mwh / data_hours, 0) if data_hours > 0 else 0
                }
            
            if fossil_peat_mwh > 0:
                renewable_sources['fossil_peat'] = {
                    'penetration_percent': round(fossil_peat_pct, 1),
                    'annual_energy_gwh': round(fossil_peat_mwh / 1000, 0),
                    'avg_power_mw': round(fossil_peat_mwh / data_hours, 0) if data_hours > 0 else 0
                }
            
            if abs(net_import_pct) > 0.5:
                renewable_sources['net_imports'] = {
                    'penetration_percent': round(net_import_pct, 1),
                    'annual_energy_gwh': round(net_import_energy_mwh / 1000, 0),
                    'avg_power_mw': round(net_import_energy_mwh / data_hours, 0) if data_hours > 0 else 0
                }
            
            if abs(residual_pct) > 0.1:
                renewable_sources['residual'] = {
                    'penetration_percent': round(residual_pct, 1),
                    'annual_energy_gwh': round(residual_energy_mwh / 1000, 0),
                    'avg_power_mw': round(residual_energy_mwh / data_hours, 0) if data_hours > 0 else 0,
                    'note': 'Unaccounted or missing generation data'
                }
            
            total_renewable = solar_pct + wind_pct + hydro_pct + biomass_pct + geothermal_pct
            
            return {
                'country_code': country_code,
                'load_column': load_col,
                'total_annual_load_gwh': round(total_load_energy_mwh / 1000, 0),
                'avg_load_mw': round(avg_load_mw, 0),
                'data_hours': round(data_hours, 1),
                'data_completeness': f"{len(load_series)/len(df)*100:.1f}%",
                'net_imports_gwh': round(net_import_energy_mwh / 1000, 0),
                'renewable_sources': renewable_sources,
                'total_renewable_percent': round(total_renewable, 1),
                'total_nuclear_percent': round(nuclear_pct, 1),
                'total_fossil_percent': round(fossil_coal_pct + fossil_gas_pct + fossil_oil_pct + fossil_peat_pct, 1),
                'identified_columns': identified_cols,
                'delta_hours': delta_hours
            }
            
        except Exception as e:
            print(f" ERROR: Analysis failed for {country_code}: {e}")
            return None
    
    def calculate_emission_reductions(self, df: pd.DataFrame, country_code: str, emission_factors: Optional[Dict[str, float]] = None) -> Dict:
        analysis = self.analyze_renewable_integration(df, country_code)
        if not analysis:
            return {}
        
        factors = self.default_emission_factors.copy()
        if emission_factors:
            factors.update(emission_factors)
        
        current_emissions = 0
        renewable_emissions = 0
        
        for source, data in analysis['renewable_sources'].items():
            if 'annual_energy_gwh' in data:
                energy_gwh = data['annual_energy_gwh']
                if source in factors:
                    if source.startswith('fossil'):
                        current_emissions += energy_gwh * factors[source]
                    elif source in ['solar', 'wind', 'hydro', 'biomass', 'geothermal', 'nuclear']:
                        renewable_emissions += energy_gwh * factors[source]
        
        fossil_emissions = current_emissions
        total_emissions = current_emissions + renewable_emissions
        
        if fossil_emissions > 0:
            reduction_pct = (fossil_emissions - total_emissions) / fossil_emissions * 100
        else:
            reduction_pct = 0
        
        return {
            'current_emissions_tons': round(total_emissions, 0),
            'fossil_only_emissions_tons': round(fossil_emissions, 0),
            'avoided_emissions_tons': round(fossil_emissions - total_emissions, 0),
            'reduction_percentage': round(reduction_pct, 1),
            'breakdown': {
                'fossil_sources': round(current_emissions, 0),
                'renewable_sources': round(renewable_emissions, 0)
            }
        }
    
    def create_integration_report(self, df: pd.DataFrame, country_code: str = 'DE') -> Optional[Dict]:
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
        print(f"   Completeness:       {analysis['data_completeness']}")
        print(f"   Time resolution:    {analysis['data_hours']:.1f} hours")
        
        print(f"\n ANNUAL SUMMARY:")
        print(f"   Total Load:         {analysis['total_annual_load_gwh']:>10,.0f} GWh")
        print(f"   Net Imports:        {analysis['net_imports_gwh']:>10,.0f} GWh")
        print(f"   Average Load:       {analysis['avg_load_mw']:>10,.0f} MW")
        
        print(f"\n ENERGY MIX (percent of annual consumption):")
        print(f"   {'-'*60}")
        
        renewable_sources = analysis['renewable_sources']
        
        source_order = ['solar', 'wind', 'hydro', 'biomass', 'geothermal', 'nuclear', 
                       'fossil_coal', 'fossil_gas', 'fossil_oil', 'fossil_peat', 'net_imports', 'residual']
        
        for source in source_order:
            if source in renewable_sources:
                data = renewable_sources[source]
                pct = data['penetration_percent']
                energy = data['annual_energy_gwh']
                source_name = source.replace('_', ' ').title()
                print(f"   {source_name:18s}: {pct:5.1f}%  ({energy:>8,.0f} GWh)")
        
        print(f"   {'-'*60}")
        print(f"   Total Renewable:    {analysis['total_renewable_percent']:5.1f}%")
        print(f"   Total Nuclear:      {analysis['total_nuclear_percent']:5.1f}%")
        print(f"   Total Fossil:       {analysis['total_fossil_percent']:5.1f}%")
        
        return analysis
    
    def plot_renewable_mix(self, df: pd.DataFrame, country_code: str = 'DE', save_path: Optional[str] = None):
        analysis = self.analyze_renewable_integration(df, country_code)
        
        if analysis is None:
            print(f"\n Cannot create plot for {country_code}")
            return None
        
        renewable_sources = analysis['renewable_sources']
        
        sources = []
        energies = []
        colors = []
        
        color_map = {
            'solar': '#FFD700',
            'wind': '#87CEEB',
            'hydro': '#4169E1',
            'biomass': '#32CD32',
            'geothermal': '#9370DB',
            'nuclear': '#FFA500',
            'fossil_coal': '#8B4513',
            'fossil_gas': '#A0522D',
            'fossil_oil': '#D2691E',
            'fossil_peat': '#8B6914',
            'net_imports': '#A9A9A9',
            'residual': '#808080'
        }
        
        source_order = ['solar', 'wind', 'hydro', 'biomass', 'geothermal', 'nuclear', 
                       'fossil_coal', 'fossil_gas', 'fossil_oil', 'fossil_peat', 'net_imports', 'residual']
        
        for source in source_order:
            if source in renewable_sources:
                data = renewable_sources[source]
                if data['annual_energy_gwh'] > 0:
                    source_name = source.replace('_', ' ').title()
                    sources.append(source_name)
                    energies.append(data['annual_energy_gwh'])
                    colors.append(color_map.get(source, '#808080'))
        
        if not sources:
            print("No data to plot")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        y_pos = np.arange(len(sources))
        bars = ax.barh(y_pos, energies, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sources, fontsize=11)
        ax.set_xlabel('Annual Energy (GWh)', fontsize=12)
        ax.set_title(f'Generation by Source - {country_code}\nTotal Load: {analysis["total_annual_load_gwh"]:,.0f} GWh', 
                   fontsize=14, fontweight='bold')
        
        max_energy = max(energies) if energies else 0
        for i, bar in enumerate(bars):
            width = bar.get_width()
            pct = (energies[i] / analysis['total_annual_load_gwh']) * 100
            label = f'{width:,.0f} GWh ({pct:.1f}%)'
            ax.text(width + max_energy*0.01, bar.get_y() + bar.get_height()/2,
                   label, ha='left', va='center', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Chart saved to: {save_path}")
        
        plt.show()
        return fig
    
    def analyze_multiple_countries(self, df: pd.DataFrame, country_codes: Optional[List[str]] = None) -> Dict:
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
    
    def run_complete_analysis(self):
        print("=" * 70)
        print("RENEWABLE INTEGRATION ANALYSIS TOOL - COMPLETE VERSION")
        print("=" * 70)
        print("\nFEATURES:")
        print("   1. Automatic detection of available data sources")
        print("   2. Time series analysis from CSV data")
        print("   3. Direct analysis from 2024 ENTSO-E data")
        print("   4. Multi-country comparison")
        print("   5. Hydro, nuclear, and fossil fuel breakdown")
        print("   6. CO2 emission calculations")
        print("=" * 70)
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(base_dir, 'data')
        
        print(f"\nLooking for data in: {data_dir}")
        
        load_file_2024 = os.path.join(data_dir, 'monthly_hourly_load_values_2024.csv')
        gen_file_2024 = os.path.join(data_dir, 'monthly_domestic_values_2024.csv')
        flows_file_2024 = os.path.join(data_dir, 'physical_energy_power_flows_2024.csv')
        
        if os.path.exists(load_file_2024) and os.path.exists(gen_file_2024) and os.path.exists(flows_file_2024):
            print("\n" + "="*60)
            print("CHECKING FOR 2024 DATA...")
            print("="*60)
            print("   2024 data found! Running analysis...")
            
            result = self.analyze_from_2024_data()
            if result is not None:
                results_df, df_gen, df_load, df_flows = result
                
                if len(results_df) == 0:
                    print("No results from 2024 data. Trying CSV data...")
                else:
                    print("\n" + "="*100)
                    print("EUROPEAN RENEWABLE INTEGRATION SUMMARY 2024")
                    print("="*100)
                    print(f"\n{'Country':<6} {'Load':>12} {'Renewable':>12} {'Ren%':>6} {'Nuclear':>10} {'Nuc%':>5} {'Fossil':>10} {'Fos%':>5} {'NetImp':>10} {'Net%':>5}")
                    print("-"*100)
                    
                    for _, r in results_df.iterrows():
                        print(f"{r['Country']:<6} {r['Load_GWh']:>12,} {r['Renewable_GWh']:>12,} {r['Renewable_Pct']:>5.1f}% "
                              f"{r['Nuclear_GWh']:>10,} {r['Nuclear_Pct']:>4.1f}% {r['Fossil_GWh']:>10,} {r['Fossil_Pct']:>4.1f}% "
                              f"{r['Net_Imports_GWh']:>10,} {r['Net_Imports_Pct']:>5.1f}%")
                    
                    results_df.to_csv('renewable_integration_results_2024.csv', index=False)
                    print("\nResults saved to: renewable_integration_results_2024.csv")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    top10 = results_df.head(10)
                    colors = ['#2ecc71' if x >= 50 else '#f39c12' if x >= 30 else '#e74c3c' for x in top10['Renewable_Pct']]
                    ax.barh(top10['Country'], top10['Renewable_Pct'], color=colors)
                    ax.set_xlabel('Renewable Penetration (% of Load)', fontsize=12)
                    ax.set_title('Top 10 Countries - Renewable Energy Share 2024', fontsize=14, fontweight='bold')
                    for bar, pct in zip(ax.patches, top10['Renewable_Pct']):
                        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', ha='left', va='center', fontsize=9)
                    ax.axvline(x=50, color='green', linestyle='--', label='50% Target')
                    ax.axvline(x=30, color='orange', linestyle='--', label='30% Baseline')
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='x')
                    plt.tight_layout()
                    plt.savefig('renewable_2024_chart.png', dpi=150)
                    print("Chart saved to: renewable_2024_chart.png")
                    
                    print("\n" + "="*60)
                    print("2024 ANALYSIS COMPLETED SUCCESSFULLY")
                    print("="*60)
                    return
        
        csv_file = os.path.join(data_dir, 'europe_energy_real.csv')
        
        if os.path.exists(csv_file):
            print(f"\nFound CSV file: {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                if 'utc_timestamp' in df.columns:
                    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
                    df.set_index('utc_timestamp', inplace=True)
                    print(f"Date range: {df.index.min()} to {df.index.max()}")
                
                country_codes = ['DE', 'FR', 'IT', 'ES', 'GB', 'NL', 'PL', 'BE', 'AT']
                
                print(f"\nAnalyzing {len(country_codes)} countries...")
                all_results = self.analyze_multiple_countries(df, country_codes)
                
                if all_results:
                    print(f"\n{'='*70}")
                    print("EUROPEAN RENEWABLE INTEGRATION SUMMARY")
                    print("="*70)
                    
                    summary_data = []
                    for country_code, result in all_results.items():
                        sources = result['renewable_sources']
                        
                        summary_data.append({
                            'Country': country_code,
                            'Solar_%': sources.get('solar', {}).get('penetration_percent', 0),
                            'Wind_%': sources.get('wind', {}).get('penetration_percent', 0),
                            'Hydro_%': sources.get('hydro', {}).get('penetration_percent', 0),
                            'Biomass_%': sources.get('biomass', {}).get('penetration_percent', 0),
                            'Nuclear_%': sources.get('nuclear', {}).get('penetration_percent', 0),
                            'Total_RE_%': result['total_renewable_percent'],
                            'Load_GWh': result['total_annual_load_gwh']
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df = summary_df.sort_values('Total_RE_%', ascending=False)
                    print("\n", summary_df.to_string(index=False))
                    
                    summary_df.to_csv('renewable_integration_summary.csv', index=False)
                    print(f"\nSummary saved to: renewable_integration_summary.csv")
                    
                    first_country = list(all_results.keys())[0]
                    print(f"\nCreating detailed report for {first_country}...")
                    self.create_integration_report(df, first_country)
                    
                    print(f"\nCreating bar chart for {first_country}...")
                    self.plot_renewable_mix(df, first_country, f'{first_country}_renewable_mix.png')
                    return
                
            except Exception as e:
                print(f"Error loading CSV: {e}")
        
        print("\n" + "="*60)
        print("NO DATA FILES FOUND OR NO RESULTS")
        print("="*60)
        print("\nRunning with synthetic data...")
        
        self.run_with_synthetic_data()
    
    def run_with_synthetic_data(self):
        print("\nGenerating synthetic data for testing...")
        
        hours = 8760
        dates_hourly = pd.date_range('2024-01-01', periods=hours, freq='h')
        
        np.random.seed(42)
        n = hours
        
        time_factor = np.sin(2 * np.pi * np.arange(n) / 8760) * 0.3 + 0.7
        daily_pattern = 0.7 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 24)
        
        data = {
            'DE_load_actual': np.random.normal(55000, 5000, n) * time_factor,
            'DE_solar_generation': np.random.normal(8000, 2000, n) * daily_pattern,
            'DE_wind_generation': np.random.normal(15000, 4000, n) * (0.5 + np.random.rand(n) * 0.5),
            'DE_hydro_generation': np.random.normal(3000, 500, n) * 0.8,
            'DE_nuclear_generation': np.random.normal(6000, 200, n) * 0.9,
            'DE_biomass_generation': np.random.normal(2500, 300, n) * 0.85,
            'FR_load_actual': np.random.normal(45000, 4000, n) * time_factor,
            'FR_solar_generation': np.random.normal(5000, 1500, n) * daily_pattern,
            'FR_wind_generation': np.random.normal(10000, 3000, n) * (0.5 + np.random.rand(n) * 0.5),
            'FR_hydro_generation': np.random.normal(8000, 1000, n) * 0.9,
            'FR_nuclear_generation': np.random.normal(35000, 2000, n) * 0.95,
            'FR_biomass_generation': np.random.normal(1500, 200, n) * 0.8,
            'IT_load_actual': np.random.normal(35000, 3000, n) * time_factor,
            'IT_solar_generation': np.random.normal(7000, 1500, n) * daily_pattern,
            'IT_wind_generation': np.random.normal(5000, 1500, n) * (0.5 + np.random.rand(n) * 0.5),
            'IT_hydro_generation': np.random.normal(4000, 600, n) * 0.85,
            'ES_load_actual': np.random.normal(30000, 2500, n) * time_factor,
            'ES_solar_generation': np.random.normal(9000, 2000, n) * daily_pattern,
            'ES_wind_generation': np.random.normal(8000, 2000, n) * (0.5 + np.random.rand(n) * 0.5),
            'ES_hydro_generation': np.random.normal(3000, 500, n) * 0.8,
            'ES_nuclear_generation': np.random.normal(5000, 200, n) * 0.95,
            'GB_load_actual': np.random.normal(40000, 3500, n) * time_factor,
            'GB_wind_generation': np.random.normal(12000, 3000, n) * (0.5 + np.random.rand(n) * 0.5),
            'GB_solar_generation': np.random.normal(3000, 1000, n) * daily_pattern,
            'GB_nuclear_generation': np.random.normal(8000, 500, n) * 0.95,
        }
        
        for col in data:
            data[col] = np.maximum(data[col], 0)
        
        df = pd.DataFrame(data, index=dates_hourly)
        
        print(f"Synthetic data created: {df.shape[0]} hours, {df.shape[1]} columns")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        countries = ['DE', 'FR', 'IT', 'ES', 'GB']
        results = []
        
        for country in countries:
            print(f"\n{'='*50}")
            print(f"Analyzing {country}...")
            print(f"{'='*50}")
            
            result = self.analyze_renewable_integration(df, country)
            if result:
                results.append(result)
                
                print(f"\nRESULTS FOR {country}:")
                print(f"   Total Load: {result['total_annual_load_gwh']:,.0f} GWh")
                print(f"   Average Load: {result['avg_load_mw']:,.0f} MW")
                print(f"   Renewable Share: {result['total_renewable_percent']:.1f}%")
                print(f"   Nuclear Share: {result['total_nuclear_percent']:.1f}%")
                print(f"   Fossil Share: {result['total_fossil_percent']:.1f}%")
                
                emissions = self.calculate_emission_reductions(df, country)
                if emissions:
                    print(f"\n   CO2 EMISSIONS:")
                    print(f"      Current: {emissions['current_emissions_tons']:,.0f} tons")
                    print(f"      Avoided: {emissions['avoided_emissions_tons']:,.0f} tons")
                    print(f"      Reduction: {emissions['reduction_percentage']:.1f}%")
        
        if results:
            print(f"\n{'='*70}")
            print("EUROPEAN RENEWABLE INTEGRATION SUMMARY - SYNTHETIC DATA")
            print("="*70)
            
            summary_data = []
            for r in results:
                summary_data.append({
                    'Country': r['country_code'],
                    'Load_GWh': r['total_annual_load_gwh'],
                    'Renewable_%': r['total_renewable_percent'],
                    'Nuclear_%': r['total_nuclear_percent'],
                    'Fossil_%': r['total_fossil_percent']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Renewable_%', ascending=False)
            
            print(f"\n{'Country':<8} {'Load_GWh':>12} {'Renewable%':>10} {'Nuclear%':>10} {'Fossil%':>10}")
            print("-"*55)
            for _, row in summary_df.iterrows():
                print(f"{row['Country']:<8} {row['Load_GWh']:>12,} {row['Renewable_%']:>9.1f}% {row['Nuclear_%']:>9.1f}% {row['Fossil_%']:>9.1f}%")
            
            summary_df.to_csv('synthetic_analysis_results.csv', index=False)
            print(f"\nResults saved to: synthetic_analysis_results.csv")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            x = np.arange(len(results))
            width = 0.25
            
            renew_pcts = [r['total_renewable_percent'] for r in results]
            nuclear_pcts = [r['total_nuclear_percent'] for r in results]
            fossil_pcts = [r['total_fossil_percent'] for r in results]
            countries_list = [r['country_code'] for r in results]
            
            axes[0].bar(x - width, renew_pcts, width, label='Renewable', color='#2ecc71')
            axes[0].bar(x, nuclear_pcts, width, label='Nuclear', color='#f39c12')
            axes[0].bar(x + width, fossil_pcts, width, label='Fossil', color='#e74c3c')
            axes[0].set_xlabel('Country')
            axes[0].set_ylabel('Share (%)')
            axes[0].set_title('Energy Mix Comparison')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(countries_list)
            axes[0].legend()
            axes[0].axhline(y=50, color='green', linestyle='--', label='EU 2030 Target')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            loads = [r['total_annual_load_gwh'] / 1000 for r in results]
            axes[1].bar(countries_list, loads, color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c'])
            axes[1].set_xlabel('Country')
            axes[1].set_ylabel('Annual Load (TWh)')
            axes[1].set_title('Total Annual Load')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            for bar, load in zip(axes[1].patches, loads):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{load:.0f} TWh', ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('synthetic_analysis_chart.png', dpi=150)
            plt.show()
            print("Chart saved to: synthetic_analysis_chart.png")
            
            print("\nCreating renewable mix chart for Germany...")
            self.plot_renewable_mix(df, 'DE', 'germany_renewable_mix.png')


if __name__ == "__main__":
    analyzer = RenewableIntegrationAnalyzer()
    analyzer.run_complete_analysis()
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETED")
    print("="*70)
