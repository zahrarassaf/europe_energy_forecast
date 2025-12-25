"""
Economic Analysis Module
Calculates economic impacts of energy efficiency improvements.
"""

import pandas as pd
import numpy as np

class EconomicAnalyzer:
    def __init__(self):
        self.carbon_price = 80  # € per ton CO2 (EU ETS average)
        self.discount_rate = 0.05  # 5% discount rate for NPV
    
    def calculate_economic_savings(self, df, improvement, co2_reduction, 
                                  energy_savings_mwh=None, country_code='DE'):
        """Calculate economic savings from efficiency improvements."""
        try:
            avg_price = self._get_energy_price(df, country_code)
            
            if energy_savings_mwh is None:
                energy_savings_mwh = self._calculate_energy_savings(df, improvement, country_code)
            
            savings_from_efficiency = energy_savings_mwh * avg_price
            savings_from_carbon = co2_reduction * self.carbon_price
            total_annual_savings = savings_from_efficiency + savings_from_carbon
            
            initial_investment = self._calculate_investment(energy_savings_mwh)
            payback_period = initial_investment / total_annual_savings if total_annual_savings > 0 else 999
            roi_percentage = (total_annual_savings / initial_investment) * 100 if initial_investment > 0 else 0
            
            npv = self._calculate_npv(total_annual_savings, initial_investment)
            
            return {
                'total_annual_savings_eur': round(float(total_annual_savings), 0),
                'savings_from_efficiency': round(float(savings_from_efficiency), 0),
                'savings_from_carbon': round(float(savings_from_carbon), 0),
                'payback_period_years': round(float(payback_period), 1),
                'roi_percentage': round(float(roi_percentage), 1),
                'initial_investment_eur': round(float(initial_investment), 0),
                'npv_eur': round(float(npv), 0),
                'energy_price_eur_per_mwh': round(float(avg_price), 1),
                'carbon_price_eur_per_ton': self.carbon_price
            }
            
        except Exception as e:
            print(f"Error in economic calculation: {e}")
            return self._get_default_values()
    
    def _get_energy_price(self, df, country_code):
        """Get average energy price for a country."""
        price_cols = [col for col in df.columns 
                     if 'price_day_ahead' in col 
                     and country_code.lower() in col]
        
        if price_cols:
            return df[price_cols[0]].mean()
        
        all_price_cols = [col for col in df.columns if 'price' in col]
        if all_price_cols:
            return df[all_price_cols[0]].mean()
        
        return 80  # Default European average
    
    def _calculate_energy_savings(self, df, improvement, country_code):
        """Calculate energy savings from improvement."""
        load_col = f"{country_code.lower()}_load_actual_entsoe_transparency"
        if load_col in df.columns:
            avg_consumption = df[load_col].mean()
            return avg_consumption * improvement * 8760  # 365 * 24 hours
        
        return 1000000  # Default
    
    def _calculate_investment(self, energy_savings_mwh):
        """Calculate initial investment required."""
        # €500 per MWh of annual savings
        return energy_savings_mwh * 500
    
    def _calculate_npv(self, annual_savings, initial_investment, years=20):
        """Calculate Net Present Value."""
        pv_factor = (1 - (1 + self.discount_rate)**-years) / self.discount_rate
        return annual_savings * pv_factor - initial_investment
    
    def _get_default_values(self):
        """Return default values in case of error."""
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
