import pandas as pd
import numpy as np

class EconomicAnalyzer:
    def __init__(self):
        # Energy prices (EUR/MWh) - source: European power exchange data
        self.energy_prices = {
            'DE': 85.5,   # Germany
            'FR': 72.3,   # France
            'IT': 98.7,   # Italy
            'ES': 68.9,   # Spain
            'UK': 91.2,   # United Kingdom
            'NL': 88.1,   # Netherlands
            'BE': 79.4,   # Belgium
            'PL': 76.8    # Poland
        }
        
        # Carbon price (EUR/ton CO2) - EU ETS
        self.carbon_price = 65.0
    
    def calculate_economic_savings(self, df, improvement_percentage, carbon_reduction_tons, country='DE'):
        """Calculate economic savings from forecasting improvement"""
        
        if f'{country}_load_actual_entsoe_transparency' not in df.columns:
            return None
        
        avg_load = df[f'{country}_load_actual_entsoe_transparency'].mean()
        energy_price = self.energy_prices.get(country, 80.0)
        
        # Operational savings from load reduction
        load_reduction_percentage = improvement_percentage * 0.2  # Conservative estimate
        daily_load_reduction_mwh = avg_load * (load_reduction_percentage / 100) * 24
        
        # Energy cost savings
        daily_energy_savings = daily_load_reduction_mwh * (energy_price / 1000)  # EUR/day
        annual_energy_savings = daily_energy_savings * 365  # EUR/year
        
        # Carbon cost savings
        annual_carbon_savings = carbon_reduction_tons * self.carbon_price  # EUR/year
        
        # Total savings
        total_annual_savings = annual_energy_savings + annual_carbon_savings
        
        # Grid investment deferral (estimated)
        grid_investment_deferral = self._calculate_grid_investment_deferral(improvement_percentage, avg_load)
        
        return {
            'country': country,
            'energy_price_eur_mwh': energy_price,
            'carbon_price_eur_ton': self.carbon_price,
            'daily_load_reduction_mwh': daily_load_reduction_mwh,
            'daily_energy_savings_eur': daily_energy_savings,
            'annual_energy_savings_eur': annual_energy_savings,
            'annual_carbon_savings_eur': annual_carbon_savings,
            'total_annual_savings_eur': total_annual_savings,
            'grid_investment_deferral_eur': grid_investment_deferral,
            'payback_period_years': self._calculate_payback_period(total_annual_savings),
            'roi_percentage': self._calculate_roi(total_annual_savings)
        }
    
    def _calculate_grid_investment_deferral(self, improvement_percentage, avg_load):
        """Estimate grid investment deferral from improved forecasting"""
        # Based on industry studies: 1% load reduction can defer $1M investment per 100MW
        investment_deferral_rate = 10000  # EUR per MW per percentage point
        return improvement_percentage * avg_load * investment_deferral_rate
    
    def _calculate_payback_period(self, annual_savings):
        """Calculate payback period for ML implementation"""
        implementation_cost = 500000  # Estimated ML system implementation cost
        return implementation_cost / annual_savings if annual_savings > 0 else float('inf')
    
    def _calculate_roi(self, annual_savings):
        """Calculate return on investment"""
        implementation_cost = 500000
        return (annual_savings / implementation_cost) * 100
