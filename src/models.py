import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


# =========================================================
# Utility functions
# =========================================================

def series_to_supervised(data, n_in=24, n_out=1):
    """
    Convert time-series into supervised learning format
    for LSTM models.
    """
    df = pd.DataFrame(data)
    cols, names = [], []

    # input sequence (t-n ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f"var(t-{i})_{j}" for j in range(df.shape[1])]

    # forecast sequence (t)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        names += [f"var(t+{i})_{j}" for j in range(df.shape[1])]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg


# =========================================================
# ARIMA Model
# =========================================================

class ARIMAModel:
    def __init__(self, order=(5, 1, 2)):
        self.order = order
        self.model = None

    def fit(self, train_series):
        """Train ARIMA model."""
        self.model = ARIMA(train_series, order=self.order).fit()

    def predict(self, steps=1):
        """Forecast future values."""
        return self.model.forecast(steps=steps)


# =========================================================
# LSTM Model
# =========================================================

class LSTMModel:
    def __init__(self, input_dim, neurons=64, n_in=24):
        self.neurons = neurons
        self.n_in = n_in
        self.model = Sequential([
            LSTM(neurons, activation="tanh", input_shape=(n_in, input_dim)),
            Dense(input_dim)
        ])
        self.model.compile(optimizer="adam", loss="mse")

    def fit(self, X_train, y_train, epochs=20, batch_size=32):
        """Train the LSTM model with early stopping."""
        es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[es]
        )

    def predict(self, X_input):
        """Predict using LSTM."""
        return self.model.predict(X_input)


# =========================================================
# Wrapper Model combining ARIMA and LSTM
# =========================================================

class ForecastingModel:
    def __init__(self, use_lstm=True, lstm_neurons=64, history_hours=24):
        self.use_lstm = use_lstm
        self.history_hours = history_hours
        self.lstm_model = None
        self.arima_model = None

    def fit(self, df, target_column):
        """
        Train both ARIMA + LSTM on selected target.
        """

        # ============================
        # 1. ARIMA
        # ============================
        ts = df[target_column]
        self.arima_model = ARIMAModel(order=(5, 1, 2))
        self.arima_model.fit(ts)

        # ============================
        # 2. LSTM
        # ============================
        if self.use_lstm:

            supervised = series_to_supervised(df[[target_column]].values,
                                              n_in=self.history_hours)

            values = supervised.values
            X, y = values[:, :-1], values[:, -1]

            X = X.reshape((X.shape[0], self.history_hours, 1))

            self.lstm_model = LSTMModel(input_dim=1,
                                        neurons=lstm_neurons,
                                        n_in=self.history_hours)
            self.lstm_model.fit(X, y)

    def predict(self, df, target_column):
        """
        Run ARIMA + optionally LSTM for future prediction.
        """

        # 1. ARIMA Forecast
        arima_pred = float(self.arima_model.predict(steps=1))

        # 2. LSTM Forecast
        if self.use_lstm:
            last_window = df[target_column].values[-self.history_hours:]
            X_input = last_window.reshape((1, self.history_hours, 1))
            lstm_pred = float(self.lstm_model.predict(X_input)[0])
            return {
                "ARIMA": arima_pred,
                "LSTM": lstm_pred
            }

        return {"ARIMA": arima_pred}
