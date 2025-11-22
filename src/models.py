import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_arima(series: pd.Series, order=(5,1,0)):
    """
    Train an ARIMA model on a univariate time series.
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted

def forecast_arima(model, steps: int = 24):
    """
    Forecast future values using a trained ARIMA model.
    """
    return model.forecast(steps=steps)

def train_lstm(X_train, y_train, units=64, epochs=20, batch_size=32):
    """
    Train an LSTM neural network for time-series forecasting.
    """
    model = Sequential()
    model.add(LSTM(units, activation='tanh', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model
