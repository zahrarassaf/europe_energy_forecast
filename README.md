# Europe Renewable Energy Forecasting Project

This project focuses on forecasting renewable energy generation across Europe using time-series methods (ARIMA, LSTM).  
The goal is to improve energy resource allocation and evaluate potential reduction in energy waste.

## Project Structure

europe_energy_forecast/
├─ data/
│ ├─ raw/ # Original unprocessed datasets
│ └─ processed/ # Cleaned, filtered datasets
├─ notebooks/
│ └─ 01_EDA_and_Modeling.ipynb # Exploratory analysis + modeling
├─ src/
│ ├─ init.py
│ ├─ io_utils.py # Data loading utilities
│ ├─ preprocessing.py # Cleaning and preprocessing
│ ├─ features.py # Feature engineering
│ ├─ models.py # ARIMA + LSTM models
│ ├─ evaluation.py # Model evaluation utilities
│ └─ run_pipeline.py # Training pipeline
├─ requirements.txt
├─ README.md
└─ LICENSE

## Dataset

The dataset comes from OPSD (Open Power System Data).  
It includes hourly electricity load, wind/solar generation, and price data across European countries.

## Goals

- Load and clean the full OPSD dataset  
- Build ARIMA and LSTM forecasting models  
- Compare model performance (RMSE, MAPE)  
- Analyze potential reduction/increase in energy waste  
- Build a reproducible forecasting pipeline  

## How to Run

1. Install requirements:
pip install -r requirements.txt


2. Start Jupyter Notebook:


jupyter notebook


3. Run the notebook inside `notebooks/`.

4. To run the full training pipeline:


python src/run_pipeline.py


## License

MIT License
