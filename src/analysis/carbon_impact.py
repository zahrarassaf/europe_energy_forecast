import pandas as pd
import numpy as np

class CarbonImpactAnalyzer:
    def __init__(self):
        # EU average carbon intensity (gCO2/kWh) - source: European Environment Agency
        self.carbon_intensity = {
            'DE': 385,   # Germany
            'FR': 52,    # France (nuclear)
            'IT': 318,   # Italy
            'ES': 204,   # Spain
            'UK': 212,   # United Kingdom
            'NL': 362,   # Netherlands
            'BE': 154,   # Belgium
            'PL': 724    # Poland (coal-heavy)
        }
    
    def calculate_carbon_reduction(self, df, improvement_percentage, target_country='DE'):
        """Calculate CO2 reduction from forecasting improvement"""
        
        # Average load reduction from better forecasting
        load_reduction_percentage = improvement_percentage * 0.3  # 30% of accuracy improvement translates to load reduction
        
        if f'{target_country}_load_actual_entsoe_transparency' not in df.columns:
            return None
        
        avg_load = df[f'{target_country}_load_actual_entsoe_transparency'].mean()
        carbon_intensity = self.carbon_intensity.get(target_country, 300)
        
        # Calculate daily load reduction
        daily_load_reduction = avg_load * (load_reduction_percentage / 100) * 24  # MWh/day
        
        # Calculate CO2 reduction
        daily_co2_reduction = daily_load_reduction * carbon_intensity / 1000  # tons CO2/day
        annual_co2_reduction = daily_co2_reduction * 365  # tons CO2/year
        
        return {
            'country': target_country,
            'improvement_percentage': improvement_percentage,
            'load_reduction_percentage': load_reduction_percentage,
            'average_load_mw': avg_load,
            'carbon_intensity_gco2_kwh': carbon_intensity,
            'daily_load_reduction_mwh': daily_load_reduction,
            'daily_co2_reduction_tons': daily_co2_reduction,
            'annual_co2_reduction_tons': annual_co2_reduction,
            'equivalent_cars_removed': annual_co2_reduction / 4.6,  # Average car emits 4.6 tons CO2/year
            'equivalent_trees_planted': annual_co2_reduction / 0.022  # Average tree absorbs 22kg CO2/year
        }
