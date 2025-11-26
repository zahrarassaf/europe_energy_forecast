# Add these imports
from src.analysis.carbon_impact import CarbonImpactAnalyzer
from src.analysis.renewable_integration import RenewableIntegrationAnalyzer
from src.analysis.economic_analysis import EconomicAnalyzer

def main():
    # ... previous code ...
    
    # 6. Environmental and Economic Impact Analysis
    print("\n6. Calculating Environmental and Economic Impact...")
    
    # Carbon impact
    carbon_analyzer = CarbonImpactAnalyzer()
    carbon_impact = carbon_analyzer.calculate_carbon_reduction(df, improvement)
    
    # Renewable integration
    renewable_analyzer = RenewableIntegrationAnalyzer()
    renewable_analysis = renewable_analyzer.analyze_renewable_integration(df, 'DE')
    
    # Economic analysis
    economic_analyzer = EconomicAnalyzer()
    economic_impact = economic_analyzer.calculate_economic_savings(
        df, improvement, 
        carbon_impact['annual_co2_reduction_tons'] if carbon_impact else 0
    )
    
    # 7. Display Comprehensive Results
    print(f"\n" + "=" * 60)
    print(f"ENVIRONMENTAL & ECONOMIC IMPACT ANALYSIS")
    print("=" * 60)
    
    if carbon_impact:
        print(f"Carbon Reduction (Germany):")
        print(f"   Annual CO2 reduction: {carbon_impact['annual_co2_reduction_tons']:,.0f} tons")
        print(f"   Equivalent to: {carbon_impact['equivalent_cars_removed']:,.0f} cars removed")
        print(f"   Or: {carbon_impact['equivalent_trees_planted']:,.0f} trees planted")
    
    if economic_impact:
        print(f"Economic Impact:")
        print(f"   Annual savings: €{economic_impact['total_annual_savings_eur']:,.0f}")
        print(f"   Payback period: {economic_impact['payback_period_years']:.1f} years")
        print(f"   ROI: {economic_impact['roi_percentage']:.1f}%")
    
    if renewable_analysis:
        print(f"Renewable Integration:")
        for source, data in renewable_analysis['renewable_sources'].items():
            print(f"   {source.title()}: {data['penetration_percentage']:.1f}% penetration")
