import pandas as pd
import numpy as np

class RenewableIntegrationAnalyzer:
    def __init__(self):
        self.renewable_columns = [
            'solar_generation_actual',
            'wind_onshore_generation_actual', 
            'wind_offshore_generation_actual',
            'wind_generation_actual'
        ]
    
    def analyze_renewable_integration(self, df, country='DE'):
        """Analyze renewable energy integration potential"""
        
        results = {}
        
        # Find renewable generation columns for the country
        solar_col = f'{country}_solar_generation_actual'
        wind_col = f'{country}_wind_generation_actual'
        wind_onshore_col = f'{country}_wind_onshore_generation_actual'
        
        # Calculate renewable penetration
        load_col = f'{country}_load_actual_entsoe_transparency'
        
        if load_col not in df.columns:
            return None
        
        total_consumption = df[load_col].sum()
        
        renewable_data = {}
        
        # Solar analysis
        if solar_col in df.columns:
            solar_generation = df[solar_col].sum()
            solar_penetration = (solar_generation / total_consumption) * 100
            renewable_data['solar'] = {
                'total_generation_mwh': solar_generation,
                'penetration_percentage': solar_penetration,
                'capacity_factor': self._calculate_capacity_factor(df, solar_col, country)
            }
        
        # Wind analysis  
        if wind_col in df.columns:
            wind_generation = df[wind_col].sum()
            wind_penetration = (wind_generation / total_consumption) * 100
            renewable_data['wind'] = {
                'total_generation_mwh': wind_generation,
                'penetration_percentage': wind_penetration,
                'capacity_factor': self._calculate_capacity_factor(df, wind_col, country)
            }
        
        # Forecasting improvement impact on renewable integration
        improved_renewable_utilization = self._calculate_utilization_improvement(renewable_data)
        
        return {
            'country': country,
            'total_consumption_mwh': total_consumption,
            'renewable_sources': renewable_data,
            'improved_utilization_percentage': improved_renewable_utilization,
            'potential_additional_renewables_gwh': improved_renewable_utilization * total_consumption / 100000
        }
    
    def _calculate_capacity_factor(self, df, gen_col, country):
        """Calculate capacity factor for renewable source"""
        if gen_col not in df.columns:
            return 0
        
        # Estimate installed capacity based on country (MW)
        estimated_capacity = {
            'DE': {'solar': 50000, 'wind': 60000},
            'FR': {'solar': 12000, 'wind': 17000},
            'ES': {'solar': 15000, 'wind': 28000}
        }
        
        source_type = 'solar' if 'solar' in gen_col else 'wind'
        capacity = estimated_capacity.get(country, {}).get(source_type, 10000)
        
        avg_generation = df[gen_col].mean()
        capacity_factor = (avg_generation / capacity) * 100 if capacity > 0 else 0
        
        return capacity_factor
