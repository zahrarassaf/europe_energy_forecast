import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class HybridEnergyForecaster:
    def __init__(self):
        self.models = {}
        self.model_predictions = {}
        self.meta_model = None
        
    def train_base_models(self, X_train, y_train, X_test, y_test):
        print("Training base models...")
        
        self.models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['gradient_boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        self.models['lightgbm'] = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            self.model_predictions[name] = test_pred
            
            print(f"{name:.<20} Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
    
    def train_meta_model(self, X_train, y_train, X_test, y_test):
        print("\nTraining meta-model...")
        
        base_predictions = np.column_stack([pred for pred in self.model_predictions.values()])
        
        self.meta_model = LinearRegression()
        self.meta_model.fit(base_predictions, y_test)
        
        final_predictions = self.meta_model.predict(base_predictions)
        final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
        final_mae = mean_absolute_error(y_test, final_predictions)
        
        print(f"Hybrid Model RMSE: {final_rmse:.2f}")
        print(f"Hybrid Model MAE: {final_mae:.2f}")
        
        return final_predictions
    
    def predict(self, X):
        base_predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            base_predictions.append(pred)
        
        base_predictions = np.column_stack(base_predictions)
        final_predictions = self.meta_model.predict(base_predictions)
        
        return final_predictions

class ARIMAModel:
    def __init__(self, order=(1,1,1)):
        self.order = order
        self.model = None
        
    def fit(self, series):
        self.model = ARIMA(series, order=self.order)
        self.fitted_model = self.model.fit()
        return self.fitted_model
    
    def predict(self, steps=30):
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

class ProphetModel:
    def __init__(self):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
    def fit(self, df):
        self.model.fit(df)
        
    def predict(self, periods=30):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast

if __name__ == "__main__":
    dates = pd.date_range('2015-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    n_samples = len(dates)
    
    X = np.random.randn(n_samples, 5)
    y = 100 + 10 * X[:, 0] + 5 * X[:, 1] + np.random.normal(0, 1, n_samples)
    
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    hybrid_model = HybridEnergyForecaster()
    hybrid_model.train_base_models(X_train, y_train, X_test, y_test)
    predictions = hybrid_model.train_meta_model(X_train, y_train, X_test, y_test)
