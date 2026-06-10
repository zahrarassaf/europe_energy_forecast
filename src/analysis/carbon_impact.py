import pandas as pd
import numpy as np

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

def main():
    print("=" * 70)
    print("CARBON IMPACT ANALYSIS USING HYBRID ENSEMBLE RESULTS")
    print("=" * 70)
    
    results = {
        'FR': {'r2': 0.9948, 'load_mean': 54649, 'improvement': 62.9},
        'IT': {'r2': 0.9940, 'load_mean': 32914, 'improvement': 73.8},
        'AT': {'r2': 0.9939, 'load_mean': 7111, 'improvement': 70.3},
        'DE': {'r2': 0.9938, 'load_mean': 55752, 'improvement': 69.8},
        'FI': {'r2': 0.9931, 'load_mean': 9576, 'improvement': 52.3},
        'PL': {'r2': 0.9930, 'load_mean': 18910, 'improvement': 68.5},
        'NL': {'r2': 0.9928, 'load_mean': 12480, 'improvement': 71.0},
        'LV': {'r2': 0.9927, 'load_mean': 823, 'improvement': 71.2},
        'RO': {'r2': 0.9925, 'load_mean': 6747, 'improvement': 69.4},
        'ES': {'r2': 0.9925, 'load_mean': 28668, 'improvement': 71.9},
        'CZ': {'r2': 0.9922, 'load_mean': 7472, 'improvement': 63.3},
        'BG': {'r2': 0.9919, 'load_mean': 4348, 'improvement': 65.4},
        'HU': {'r2': 0.9916, 'load_mean': 4833, 'improvement': 68.6},
        'EE': {'r2': 0.9913, 'load_mean': 941, 'improvement': 63.2},
        'PT': {'r2': 0.9911, 'load_mean': 5660, 'improvement': 69.3},
        'SE': {'r2': 0.9910, 'load_mean': 15836, 'improvement': 52.1},
        'NO': {'r2': 0.9902, 'load_mean': 15251, 'improvement': 43.4},
        'HR': {'r2': 0.9894, 'load_mean': 1981, 'improvement': 70.2},
        'RS': {'r2': 0.9894, 'load_mean': 4496, 'improvement': 61.6},
        'BE': {'r2': 0.9890, 'load_mean': 10007, 'improvement': 63.6},
        'GR': {'r2': 0.9878, 'load_mean': 5801, 'improvement': 66.3},
        'CY': {'r2': 0.9861, 'load_mean': 493, 'improvement': 64.5},
        'LT': {'r2': 0.9859, 'load_mean': 1285, 'improvement': 61.2},
        'IE': {'r2': 0.9852, 'load_mean': 3134, 'improvement': 62.0},
        'SI': {'r2': 0.9850, 'load_mean': 1465, 'improvement': 60.5},
        'ME': {'r2': 0.9848, 'load_mean': 378, 'improvement': 58.3},
        'DK': {'r2': 0.9831, 'load_mean': 3742, 'improvement': 59.4},
        'SK': {'r2': 0.9606, 'load_mean': 3318, 'improvement': 29.3},
        'CH': {'r2': 0.9493, 'load_mean': 6758, 'improvement': 25.0},
        'UA': {'r2': 0.9383, 'load_mean': 18336, 'improvement': 38.7},
        'LU': {'r2': 0.8737, 'load_mean': 487, 'improvement': 2.7}
    }
    
    carbon_analyzer = CarbonImpactAnalyzer()
    economic_analyzer = EconomicAnalyzer()
    
    scenario_improvements = [0.01, 0.05, 0.10]
    scenario_results = []
    
    all_country_results = []
    
    print("\n" + "="*70)
    print("RESULTS BY COUNTRY (5% IMPROVEMENT SCENARIO)")
    print("="*70)
    
    for country, data in results.items():
        improvement_pct = data['improvement']
        load_mean = data['load_mean']
        
        adj_improvement = improvement_pct / 100
        
        carbon = carbon_analyzer.calculate_carbon_reduction(load_mean, adj_improvement, country)
        economic = economic_analyzer.calculate_economic_savings(
            carbon['annual_energy_savings_mwh'], 
            carbon['annual_co2_reduction_tons']
        )
        
        all_country_results.append({
            'Country': country,
            'R2': data['r2'],
            'Load_Mean_MW': load_mean,
            'Improvement_Pct': improvement_pct,
            'CO2_Reduction_Tons': carbon['annual_co2_reduction_tons'],
            'Cars_Removed': carbon['equivalent_cars_removed'],
            'Trees_Planted': carbon['equivalent_trees_planted'],
            'Energy_Savings_MWh': carbon['annual_energy_savings_mwh'],
            'Annual_Savings_EUR': economic['total_annual_savings_eur'],
            'Investment_EUR': economic['initial_investment_eur'],
            'Payback_Years': economic['payback_period_years'],
            'ROI_Pct': economic['roi_percentage'],
            'NPV_EUR': economic['npv_eur']
        })
        
        print(f"\n{country}:")
        print(f"  Model R²: {data['r2']:.4f}, Improvement: {improvement_pct:.1f}%")
        print(f"  CO2 Reduction: {carbon['annual_co2_reduction_tons']:,.0f} tons/year")
        print(f"  Annual Savings: EUR{economic['total_annual_savings_eur']:,.0f}")
        print(f"  Payback: {economic['payback_period_years']:.1f} years")
        print(f"  ROI: {economic['roi_percentage']:.1f}%")
    
    df_results = pd.DataFrame(all_country_results)
    df_results = df_results.sort_values('CO2_Reduction_Tons', ascending=False)
    
    print("\n" + "="*70)
    print("SCENARIO ANALYSIS (AT - 5% IMPROVEMENT)")
    print("="*70)
    
    at_load = results['AT']['load_mean']
    
    for scenario in scenario_improvements:
        carbon = carbon_analyzer.calculate_carbon_reduction(at_load, scenario, 'AT')
        economic = economic_analyzer.calculate_economic_savings(
            carbon['annual_energy_savings_mwh'],
            carbon['annual_co2_reduction_tons']
        )
        
        scenario_results.append({
            'Scenario': f"{scenario*100:.0f}%",
            'CO2_Reduction_Tons': carbon['annual_co2_reduction_tons'],
            'Annual_Savings_EUR': economic['total_annual_savings_eur'],
            'Payback_Years': economic['payback_period_years'],
            'ROI_Pct': economic['roi_percentage']
        })
        
        print(f"\n{scenario*100:.0f}% Improvement:")
        print(f"  CO2 Reduction: {carbon['annual_co2_reduction_tons']:,.0f} tons/year")
        print(f"  Annual Savings: EUR{economic['total_annual_savings_eur']:,.0f}")
        print(f"  Payback: {economic['payback_period_years']:.1f} years")
        print(f"  ROI: {economic['roi_percentage']:.1f}%")
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    total_co2 = df_results['CO2_Reduction_Tons'].sum()
    total_savings = df_results['Annual_Savings_EUR'].sum()
    total_investment = df_results['Investment_EUR'].sum()
    total_npv = df_results['NPV_EUR'].sum()
    
    print(f"\nTotal CO2 Reduction (31 countries): {total_co2:,.0f} tons/year")
    print(f"Total Annual Savings: EUR{total_savings:,.0f}")
    print(f"Total Investment: EUR{total_investment:,.0f}")
    print(f"Total NPV (20 years): EUR{total_npv:,.0f}")
    
    df_results.to_csv('carbon_impact_31_countries_hybrid.csv', index=False)
    pd.DataFrame(scenario_results).to_csv('carbon_impact_scenarios_at.csv', index=False)
    
    print("\n" + "="*70)
    print("TOP 10 COUNTRIES BY CO2 REDUCTION")
    print("="*70)
    print(df_results[['Country', 'CO2_Reduction_Tons', 'ROI_Pct', 'Payback_Years']].head(10).to_string(index=False))
    
    print("\nFiles saved:")
    print("  - carbon_impact_31_countries_hybrid.csv")
    print("  - carbon_impact_scenarios_at.csv")

if __name__ == "__main__":
    main()
