import subprocess
import sys

def install_requirements():
    print("Installing dashboard requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])

def run_dashboard():
    print("Starting Europe Energy Forecast Dashboard...")
    print("Open your browser at http://localhost:8501")
    subprocess.run(["streamlit", "run", "dashboard.py"])

if __name__ == "__main__":
    try:
        import streamlit
        import plotly
        run_dashboard()
    except ImportError:
        install_requirements()
        run_dashboard()
