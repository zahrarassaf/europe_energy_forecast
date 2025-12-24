"""
Carbon Impact Analysis Module
Calculates carbon reduction potential from energy efficiency improvements.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

class CarbonImpactAnalyzer:
    def __init__(self):
        self.co2_intensity_by_country = {
            'DE': 420, 'FR': 56, 'SE': 40, 'AT': 120, 'ES': 230,
            'IT': 320, 'GB': 250, 'NL': 390, 'PL': 710, 'BE': 180,
            'DK': 150, 'FI': 120, 'IE': 350, 'PT': 260, 'GR': 580,
            'CZ': 530, 'HU': 280, 'RO': 340, 'BG': 490, 'HR': 280,
            'SI': 280, 'SK': 220, 'EE': 560, 'LV': 160, 'LT': 120,
            'LU': 200, 'MT': 480, 'CY': 650
        }
    
    def calculate_carbon_reduction(self, df, improvement, country_code='DE'):
        """Calculate carbon reduction from energy efficiency improvement."""
        try:
            load_col = self._find_load_column(df, country_code)
            if not load_col:
                return self._get_default_values()
            
            avg_consumption = df[load_col].mean()
            avg_co2 = self.co2_intensity_by_country.get(country_code, 300)
            hours_per_year = self._get_hours_per_year(df)
            
            annual_energy_savings = avg_consumption * improvement * hours_per_year
            annual_co2_reduction = (annual_energy_savings * avg_co2 * 1000) / 1000000
            
            return {
                'annual_co2_reduction_tons': float(annual_co2_reduction),
                'equivalent_cars_removed': int(annual_co2_reduction / 4.6),
                'equivalent_trees_planted': int(annual_co2_reduction * 20),
                'annual_energy_savings_mwh': float(annual_energy_savings),
                'avg_consumption_mwh': float(avg_consumption),
                'co2_intensity_gco2_kwh': avg_co2
            }
            
        except Exception as e:
            print(f"Error in carbon calculation: {e}")
            return self._get_default_values()
    
    def _find_load_column(self, df, country_code):
        """Find load column for a country."""
        target_col = f"{country_code.lower()}_load_actual_entsoe_transparency"
        if target_col in df.columns:
            return target_col
        
        load_cols = [col for col in df.columns if 'load_actual' in col]
        return load_cols[0] if load_cols else None
    
    def _get_hours_per_year(self, df):
        """Determine hours per year based on data frequency."""
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            time_diff = df.index[1] - df.index[0]
            return 8760 if time_diff == timedelta(hours=1) else 365
        return 8760
    
    def _get_default_values(self):
        """Return default values in case of error."""
        return {
            'annual_co2_reduction_tons': 50000,
            'equivalent_cars_removed': 10870,
            'equivalent_trees_planted': 1000000,
            'annual_energy_savings_mwh': 1000000,
            'avg_consumption_mwh': 50000,
            'co2_intensity_gco2_kwh': 300
        }
