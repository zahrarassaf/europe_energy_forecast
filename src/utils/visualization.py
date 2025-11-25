import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ResearchVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_research_dashboard(self, data, target_country='DE'):
        """Create comprehensive research dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{target_country} Energy Consumption',
                'Seasonal Decomposition', 
                'Cross-Country Correlation',
                'Yearly Comparison'
            )
        )
        
        # Time series plot
        fig.add_trace(
            go.Scatter(x=data.index, y=data[target_country], name=target_country),
            row=1, col=1
        )
        
        # Correlation heatmap
        corr_matrix = data[['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE', 'PL']].corr()
        
        return fig
    
    def plot_model_comparison(self, results_dict):
        """Plot comparison of multiple models"""
        models = list(results_dict.keys())
        rmse_values = [results_dict[model]['rmse'] for model in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, rmse_values, color=sns.color_palette("viridis", len(models)))
        
        ax.set_ylabel('RMSE')
        ax.set_title('Model Performance Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, rmse_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
