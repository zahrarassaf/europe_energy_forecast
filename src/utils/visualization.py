import matplotlib.pyplot as plt
import seaborn as sns

class ResearchVisualizer:
    def __init__(self):
        plt.style.use('seaborn')
    
    def plot_country_comparison(self, data, countries):
        plt.figure(figsize=(12, 6))
        for country in countries:
            plt.plot(data.index, data[country], label=country, alpha=0.7)
        plt.legend()
        plt.title('Energy Consumption - Country Comparison')
        plt.show()
