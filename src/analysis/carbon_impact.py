import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import warnings
import re
import os
warnings.filterwarnings('ignore')

class CarbonImpactAnalyzer:
    def __init__(self):
        self.co2_intensity_by_country = {
            'AT': 120, 'BE': 180, 'BG': 490, 'CH': 80, 'CY': 650,
            'CZ': 530, 'DE': 420, 'DK': 150, 'EE': 560, 'ES': 230,
            'FI': 120, 'FR': 56, 'GB': 250, 'GR': 580, 'HR': 280,
            'HU': 280, 'IE': 350, 'IT': 320, 'LT': 120, 'LU': 200,
            'LV': 160, 'ME': 300, 'NL': 390, 'NO': 40, 'PL': 710,
            'PT': 260, 'RO': 340, 'RS': 400, 'SE': 40, 'SI': 280,
            'SK': 220, 'UA': 300
        }
    
    def calculate_carbon_reduction(self, df, improvement, country_code='AT', use_default_only=False):
        if improvement <= 0:
            return {
                'annual_co2_reduction_tons': 0.0,
                'equivalent_cars_removed': 0,
                'equivalent_trees_planted': 0,
                'annual_energy_savings_mwh': 0.0,
                'avg_consumption_mwh': 0.0,
                'co2_intensity_gco2_kwh': self.co2_intensity_by_country.get(country_code, 300),
                'data_source': 'zero_improvement'
            }
        
        try:
            target_col = f"{country_code}_load_actual_entsoe_transparency"
            
            if target_col not in df.columns:
                result = self._get_default_values(country_code)
                result['annual_co2_reduction_tons'] *= improvement / 0.05
                result['annual_energy_savings_mwh'] *= improvement / 0.05
                result['equivalent_cars_removed'] = int(result['annual_co2_reduction_tons'] / 4.6)
                result['equivalent_trees_planted'] = int(result['annual_co2_reduction_tons'] * 20)
                return result
            
            if df[target_col].isna().all():
                result = self._get_default_values(country_code)
                result['annual_co2_reduction_tons'] *= improvement / 0.05
                result['annual_energy_savings_mwh'] *= improvement / 0.05
                result['equivalent_cars_removed'] = int(result['annual_co2_reduction_tons'] / 4.6)
                result['equivalent_trees_planted'] = int(result['annual_co2_reduction_tons'] * 20)
                return result
            
            avg_consumption = df[target_col].mean()
            
            if pd.isna(avg_consumption) or avg_consumption == 0:
                result = self._get_default_values(country_code)
                result['annual_co2_reduction_tons'] *= improvement / 0.05
                result['annual_energy_savings_mwh'] *= improvement / 0.05
                result['equivalent_cars_removed'] = int(result['annual_co2_reduction_tons'] / 4.6)
                result['equivalent_trees_planted'] = int(result['annual_co2_reduction_tons'] * 20)
                return result
            
            avg_co2 = self.co2_intensity_by_country.get(country_code, 300)
            
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
                try:
                    time_diff = df.index[1] - df.index[0]
                    hours_per_year = 8760 if abs(time_diff.total_seconds() / 3600 - 1) < 0.1 else 8760
                except:
                    hours_per_year = 8760
            else:
                hours_per_year = 8760
            
            annual_energy_savings = avg_consumption * improvement * hours_per_year
            annual_co2_reduction = (annual_energy_savings * avg_co2 * 1000) / 1000000
            
            if annual_co2_reduction > 1000000:
                annual_co2_reduction = 1000000
            
            return {
                'annual_co2_reduction_tons': float(annual_co2_reduction),
                'equivalent_cars_removed': int(annual_co2_reduction / 4.6),
                'equivalent_trees_planted': int(annual_co2_reduction * 20),
                'annual_energy_savings_mwh': float(annual_energy_savings),
                'avg_consumption_mwh': float(avg_consumption),
                'co2_intensity_gco2_kwh': avg_co2,
                'data_source': 'real_data'
            }
            
        except Exception:
            result = self._get_default_values(country_code)
            result['annual_co2_reduction_tons'] *= improvement / 0.05
            result['annual_energy_savings_mwh'] *= improvement / 0.05
            result['equivalent_cars_removed'] = int(result['annual_co2_reduction_tons'] / 4.6)
            result['equivalent_trees_planted'] = int(result['annual_co2_reduction_tons'] * 20)
            return result
    
    def _get_default_values(self, country_code='AT'):
        avg_co2 = self.co2_intensity_by_country.get(country_code, 300)
        
        default_multipliers = {
            'DE': 1.5, 'FR': 1.2, 'IT': 1.3, 'ES': 1.1, 'GB': 1.2,
            'PL': 1.4, 'NL': 1.0, 'BE': 0.8, 'AT': 1.0,
        }
        
        multiplier = default_multipliers.get(country_code, 1.0)
        base_co2 = 50000 * multiplier
        
        return {
            'annual_co2_reduction_tons': base_co2,
            'equivalent_cars_removed': int(base_co2 / 4.6),
            'equivalent_trees_planted': int(base_co2 * 20),
            'annual_energy_savings_mwh': base_co2 * 20,
            'avg_consumption_mwh': 50000 * multiplier,
            'co2_intensity_gco2_kwh': avg_co2,
            'data_source': 'default_estimate'
        }

class EconomicAnalyzer:
    def __init__(self):
        self.carbon_price = 80
        self.discount_rate = 0.05
    
    def calculate_economic_savings(self, df, improvement, co2_reduction, 
                                  energy_savings_mwh=None, country_code='AT', 
                                  model_performance=0.0):
        if improvement <= 0:
            return {
                'total_annual_savings_eur': 0.0,
                'savings_from_efficiency': 0.0,
                'savings_from_carbon': 0.0,
                'payback_period_years': 999.0,
                'roi_percentage': 0.0,
                'initial_investment_eur': 0.0,
                'npv_eur': 0.0,
                'energy_price_eur_per_mwh': 80.0,
                'carbon_price_eur_per_ton': self.carbon_price,
                'adjusted_improvement_percent': 0.0,
                'model_performance_impact': model_performance
            }
        
        try:
            if model_performance < 0:
                adjusted_improvement = improvement * max(0, 1 + model_performance/100)
                improvement = adjusted_improvement
            
            if improvement <= 0:
                return {
                    'total_annual_savings_eur': 0.0,
                    'savings_from_efficiency': 0.0,
                    'savings_from_carbon': 0.0,
                    'payback_period_years': 999.0,
                    'roi_percentage': 0.0,
                    'initial_investment_eur': 0.0,
                    'npv_eur': 0.0,
                    'energy_price_eur_per_mwh': 80.0,
                    'carbon_price_eur_per_ton': self.carbon_price,
                    'adjusted_improvement_percent': 0.0,
                    'model_performance_impact': model_performance
                }
            
            if pd.isna(co2_reduction) or co2_reduction <= 0:
                co2_reduction = 50000 * (improvement / 0.05)
            
            price_cols = [col for col in df.columns if 'price_day_ahead' in col and country_code in col]
            
            if price_cols:
                try:
                    avg_price = df[price_cols[0]].replace([np.inf, -np.inf], np.nan).dropna().mean()
                    if pd.isna(avg_price) or avg_price <= 0:
                        avg_price = 80
                except:
                    avg_price = 80
            else:
                avg_price = 80
            
            if energy_savings_mwh is None or pd.isna(energy_savings_mwh) or energy_savings_mwh <= 0:
                energy_savings_mwh = 1000000 * (improvement / 0.05)
            
            savings_from_efficiency = energy_savings_mwh * avg_price
            savings_from_carbon = co2_reduction * self.carbon_price
            total_annual_savings = savings_from_efficiency + savings_from_carbon
            
            initial_investment = energy_savings_mwh * 500
            
            if total_annual_savings > 0 and initial_investment > 0:
                payback_period = initial_investment / total_annual_savings
                roi_percentage = (total_annual_savings / initial_investment) * 100
            else:
                payback_period = 999.0
                roi_percentage = 0.0
            
            if total_annual_savings > 0:
                npv = total_annual_savings * ((1 - (1 + self.discount_rate)**-20) / self.discount_rate) - initial_investment
            else:
                npv = -initial_investment
            
            return {
                'total_annual_savings_eur': round(float(total_annual_savings), 0),
                'savings_from_efficiency': round(float(savings_from_efficiency), 0),
                'savings_from_carbon': round(float(savings_from_carbon), 0),
                'payback_period_years': round(float(payback_period), 1),
                'roi_percentage': round(float(roi_percentage), 1),
                'initial_investment_eur': round(float(initial_investment), 0),
                'npv_eur': round(float(npv), 0),
                'energy_price_eur_per_mwh': round(float(avg_price), 1),
                'carbon_price_eur_per_ton': self.carbon_price,
                'adjusted_improvement_percent': improvement * 100,
                'model_performance_impact': model_performance
            }
            
        except Exception as e:
            print(f"Economic calculation error for {country_code}: {e}")
            return {
                'total_annual_savings_eur': 0.0,
                'savings_from_efficiency': 0.0,
                'savings_from_carbon': 0.0,
                'payback_period_years': 999.0,
                'roi_percentage': 0.0,
                'initial_investment_eur': 0.0,
                'npv_eur': 0.0,
                'energy_price_eur_per_mwh': 80.0,
                'carbon_price_eur_per_ton': self.carbon_price,
                'adjusted_improvement_percent': 0.0,
                'model_performance_impact': model_performance
            }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.output_layer(x).squeeze()

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=168):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        return (
            self.X[idx:idx+self.sequence_length], 
            self.y[idx+self.sequence_length]
        )

class ScientificEnergyPredictor:
    def __init__(self, sequence_length=168, country='AT'):
        self.sequence_length = sequence_length
        self.country = country
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.carbon_analyzer = CarbonImpactAnalyzer()
        self.economic_analyzer = EconomicAnalyzer()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.original_data = None
        self.training_losses = []
        self.val_losses = []
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.y_train_original = None
        self.y_val_original = None
        self.y_test_original = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.input_dim = None
    
    def prepare_features_scientifically(self, df, target_col):
        features = []
        
        lags = [1, 2, 3, 4, 5, 6, 7, 24, 48, 72, 168]
        for lag in lags:
            if len(df) > lag:
                shifted = df[target_col].shift(lag).values.reshape(-1, 1)
                features.append(shifted)
        
        if 'utc_timestamp' in df.columns:
            timestamps = pd.to_datetime(df['utc_timestamp'])
            
            hour_sin = np.sin(2 * np.pi * timestamps.dt.hour / 24)
            hour_cos = np.cos(2 * np.pi * timestamps.dt.hour / 24)
            weekday_sin = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
            weekday_cos = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
            month_sin = np.sin(2 * np.pi * timestamps.dt.month / 12)
            month_cos = np.cos(2 * np.pi * timestamps.dt.month / 12)
            
            features.extend([
                hour_sin.values.reshape(-1, 1),
                hour_cos.values.reshape(-1, 1),
                weekday_sin.values.reshape(-1, 1),
                weekday_cos.values.reshape(-1, 1),
                month_sin.values.reshape(-1, 1),
                month_cos.values.reshape(-1, 1)
            ])
        
        if len(df) > 24:
            rolling_mean_24h = df[target_col].rolling(window=24, min_periods=1).mean().values.reshape(-1, 1)
            rolling_std_24h = df[target_col].rolling(window=24, min_periods=1).std().values.reshape(-1, 1)
            features.extend([rolling_mean_24h, rolling_std_24h])
        
        features = np.hstack(features)
        
        for i in range(features.shape[1]):
            col = features[:, i]
            mask = np.isnan(col)
            if mask.any():
                first_valid = np.where(~mask)[0]
                if len(first_valid) > 0:
                    col[mask] = col[first_valid[0]]
                features[:, i] = col
        
        return features
    
    def load_and_prepare_data(self, filepath='data/europe_energy_real.csv'):
        print(f"Loading data for {self.country} from: {filepath}")
        df = pd.read_csv(filepath)
        
        if 'utc_timestamp' in df.columns:
            df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
            df.set_index('utc_timestamp', inplace=True)
        
        target_col = f"{self.country}_load_actual_entsoe_transparency"
        
        if target_col not in df.columns:
            possible_targets = [
                f"{self.country}_load_actual_entsoe_transparency",
                f"{self.country.lower()}_load_actual_entsoe_transparency",
                f"{self.country}_load_actual",
                f"{self.country}_load"
            ]
            
            for possible in possible_targets:
                if possible in df.columns:
                    target_col = possible
                    break
            
            if target_col == f"{self.country}_load_actual_entsoe_transparency":
                print(f"Target column for {self.country} not found, skipping...")
                return None, None, None
        
        print(f"Using target column: {target_col}")
        
        X = self.prepare_features_scientifically(df, target_col)
        y = df[target_col].values
        
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 1000:
            print(f"Insufficient data for {self.country}: {len(X)} samples, skipping...")
            return None, None, None
        
        print(f"Clean data shape: X={X.shape}, y={y.shape}")
        
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        total_samples = len(X_scaled)
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)
        
        self.X_train = X_scaled[:train_end]
        self.y_train = y_scaled[:train_end]
        self.y_train_original = y[:train_end]
        
        self.X_val = X_scaled[train_end:val_end]
        self.y_val = y_scaled[train_end:val_end]
        self.y_val_original = y[train_end:val_end]
        
        self.X_test = X_scaled[val_end:]
        self.y_test = y_scaled[val_end:]
        self.y_test_original = y[val_end:]
        
        print(f"Train: {len(self.X_train)} samples (70%)")
        print(f"Validation: {len(self.X_val)} samples (15%)")
        print(f"Test: {len(self.X_test)} samples (15%)")
        
        train_dataset = TimeSeriesDataset(self.X_train, self.y_train, self.sequence_length)
        val_dataset = TimeSeriesDataset(self.X_val, self.y_val, self.sequence_length)
        test_dataset = TimeSeriesDataset(self.X_test, self.y_test, self.sequence_length)
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.input_dim = self.X_train.shape[1]
        self.original_data = df.iloc[valid_idx].copy()
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def build_model(self):
        self.model = TimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=64,
            nhead=8,
            num_layers=3,
            dropout=0.1
        ).to(self.device)
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"Model built: {params:,} parameters")
        return self.model
    
    def train(self, epochs=10, lr=0.0005, patience=5):
        if self.model is None:
            self.build_model()
        
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        self.training_losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            self.training_losses.append(avg_train_loss)
            
            avg_val_loss = self._validate(criterion, self.val_loader)
            self.val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} *")
            else:
                patience_counter += 1
                print(f"Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return self.training_losses, self.val_losses
    
    def _validate(self, criterion, loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def evaluate(self, loader, set_name="Test"):
        if self.model is None:
            print("Model not trained")
            return None, None, None
        
        self.model.eval()
        predictions_scaled = []
        actuals_scaled = []
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                preds = self.model(X_batch)
                predictions_scaled.extend(preds.cpu().numpy())
                actuals_scaled.extend(y_batch.cpu().numpy())
        
        predictions_scaled = np.array(predictions_scaled)
        actuals_scaled = np.array(actuals_scaled)
        
        predictions = self.target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        if loader == self.val_loader:
            actuals = self.y_val_original[-len(predictions):]
        elif loader == self.test_loader:
            actuals = self.y_test_original[-len(predictions):]
        else:
            raise ValueError("Unknown loader")
        
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        persistence_forecast = np.roll(actuals, 24)
        valid_idx = ~np.isnan(persistence_forecast)
        persistence_mae = np.mean(np.abs(actuals[valid_idx] - persistence_forecast[valid_idx]))
        
        error_reduction = (persistence_mae - mae) / persistence_mae * 100 if persistence_mae > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"EVALUATION ON {set_name.upper()} SET FOR {self.country}")
        print(f"{'='*60}")
        print(f"RMSE: {rmse:.2f} MW")
        print(f"MAE: {mae:.2f} MW")
        print(f"24h Persistence MAE: {persistence_mae:.2f} MW")
        print(f"Improvement over persistence: {error_reduction:.1f}%")
        
        return predictions, actuals, error_reduction
    
    def forecast_future_scientifically(self, n_days=3, use_teacher_forcing=True):
        if self.model is None:
            print("Model not trained")
            return None
        
        self.model.eval()
        
        print(f"\nGenerating {n_days}-day forecast for {self.country}...")
        
        total_test_samples = len(self.X_test)
        start_idx = total_test_samples - self.sequence_length
        
        if start_idx < 0:
            start_idx = 0
            current_sequence = np.zeros((self.sequence_length, self.input_dim))
            available_len = len(self.X_test)
            current_sequence[-available_len:] = self.X_test[:available_len]
        else:
            current_sequence = self.X_test[start_idx:start_idx + self.sequence_length].copy()
        
        forecasts = []
        timestamps = []
        last_timestamp = pd.Timestamp.now()
        
        with torch.no_grad():
            for i in range(n_days * 24):
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                pred_scaled_tensor = self.model(input_tensor)
                pred_scaled = pred_scaled_tensor.item()
                
                pred = self.target_scaler.inverse_transform(
                    np.array([[pred_scaled]])
                ).flatten()[0]
                
                forecasts.append(pred)
                timestamps.append(last_timestamp + timedelta(hours=i+1))
                
                current_sequence = np.roll(current_sequence, -1, axis=0)
                
                if use_teacher_forcing:
                    teacher_idx = start_idx + self.sequence_length + i
                    if teacher_idx < len(self.y_test_original):
                        next_actual = self.y_test_original[teacher_idx]
                        next_actual_scaled = self.target_scaler.transform(
                            np.array([[next_actual]])
                        ).flatten()[0]
                        current_sequence[-1, 0] = next_actual_scaled
                    else:
                        current_sequence[-1, 0] = pred_scaled
                else:
                    current_sequence[-1, 0] = pred_scaled
        
        forecast_df = pd.DataFrame({
            'timestamp': timestamps,
            'predicted_load': forecasts
        })
        
        forecast_csv_path = f'scientific_forecast_{self.country}_{n_days}days.csv'
        forecast_df.to_csv(forecast_csv_path, index=False)
        
        print(f"\nForecast saved to: {forecast_csv_path}")
        
        return forecast_df
    
    def _check_country_data_availability(self, country_code):
        target_col = f"{country_code}_load_actual_entsoe_transparency"
        
        if self.original_data is not None and target_col in self.original_data.columns:
            data = self.original_data[target_col]
            if not data.isna().all() and data.mean() > 0:
                return True
        return False
    
    def analyze_carbon_impact_with_uncertainty(self, error_reduction_percent, 
                                              improvement_scenarios=[0.01, 0.05, 0.10]):
        print(f"\n{'='*60}")
        print(f"CARBON IMPACT ANALYSIS FOR {self.country}")
        print(f"{'='*60}")
        
        if self.original_data is None:
            print("No data available")
            return {}
        
        print(f"Model Performance: {error_reduction_percent:.1f}% vs persistence")
        
        if error_reduction_percent < 0:
            print(f"WARNING: Model performs worse than baseline")
            adjusted_scenarios = []
            for scenario in improvement_scenarios:
                if error_reduction_percent < -10:
                    adjusted = 0
                else:
                    adjusted = scenario * max(0, 1 + error_reduction_percent/100)
                adjusted_scenarios.append(adjusted)
            improvement_scenarios = adjusted_scenarios
        
        scenario_results = {}
        country_codes = list(self.carbon_analyzer.co2_intensity_by_country.keys())
        
        for scenario_idx, improvement in enumerate(improvement_scenarios):
            print(f"\nScenario {scenario_idx+1}: {improvement*100:.1f}% improvement")
            print("-" * 50)
            
            scenario_co2 = 0
            scenario_energy = 0
            
            for country_code in country_codes:
                has_real_data = self._check_country_data_availability(country_code)
                
                carbon_result = self.carbon_analyzer.calculate_carbon_reduction(
                    self.original_data, improvement, country_code, use_default_only=not has_real_data
                )
                
                if scenario_idx == 0:
                    data_source = "REAL" if carbon_result.get('data_source') == 'real_data' else "ESTIMATE"
                    print(f"{country_code}: {carbon_result['annual_co2_reduction_tons']:,.0f} tons ({data_source})")
                
                scenario_co2 += carbon_result['annual_co2_reduction_tons']
                scenario_energy += carbon_result['annual_energy_savings_mwh']
            
            scenario_results[f'{improvement*100:.0f}pct'] = {
                'improvement_percent': improvement * 100,
                'total_co2_reduction_tons': scenario_co2,
                'total_energy_savings_mwh': scenario_energy,
                'model_performance': error_reduction_percent
            }
            
            print(f"Total CO2 Reduction: {scenario_co2:,.0f} tons")
            print(f"Total Energy Savings: {scenario_energy:,.0f} MWh")
        
        scenario_df = pd.DataFrame(scenario_results).T
        scenario_df.to_csv(f'carbon_impact_scenarios_{self.country}.csv')
        print(f"\nResults saved to: carbon_impact_scenarios_{self.country}.csv")
        
        return scenario_results
    
    def analyze_economic_impact(self, improvement=0.05, model_performance=0.0):
        print(f"\n{'='*60}")
        print(f"ECONOMIC IMPACT ANALYSIS FOR {self.country}")
        print(f"{'='*60}")
        
        if self.original_data is None:
            print("No data available")
            return {}
        
        print(f"\nAssumptions:")
        print(f"  Base Efficiency Improvement: {improvement*100:.1f}%")
        print(f"  Model Performance: {model_performance:.1f}%")
        print(f"  Carbon Price: EUR{self.economic_analyzer.carbon_price}/ton")
        print(f"  Discount Rate: {self.economic_analyzer.discount_rate*100:.1f}%")
        
        results_by_country = {}
        country_codes = list(self.carbon_analyzer.co2_intensity_by_country.keys())
        
        total_annual_savings = 0
        total_investment = 0
        total_npv = 0
        
        print(f"\n{'='*60}")
        print("RESULTS BY COUNTRY")
        print(f"{'='*60}")
        
        for country_code in country_codes:
            has_real_data = self._check_country_data_availability(country_code)
            
            carbon_result = self.carbon_analyzer.calculate_carbon_reduction(
                self.original_data, improvement, country_code, use_default_only=not has_real_data
            )
            
            economic_result = self.economic_analyzer.calculate_economic_savings(
                self.original_data,
                improvement,
                carbon_result['annual_co2_reduction_tons'],
                carbon_result['annual_energy_savings_mwh'],
                country_code,
                model_performance
            )
            
            results_by_country[country_code] = {
                'carbon': carbon_result,
                'economic': economic_result,
                'has_real_data': has_real_data
            }
            
            print(f"\n{country_code}:")
            print(f"  Annual Savings: EUR{economic_result['total_annual_savings_eur']:,}")
            print(f"  Investment: EUR{economic_result['initial_investment_eur']:,}")
            print(f"  Payback: {economic_result['payback_period_years']} years")
            print(f"  ROI: {economic_result['roi_percentage']:.1f}%")
            
            total_annual_savings += economic_result['total_annual_savings_eur']
            total_investment += economic_result['initial_investment_eur']
            total_npv += economic_result['npv_eur']
        
        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print(f"{'='*60}")
        
        if total_annual_savings > 0:
            avg_payback = total_investment / total_annual_savings
            avg_roi = (total_annual_savings / total_investment) * 100
        else:
            avg_payback = 999
            avg_roi = 0
        
        print(f"Total Annual Savings: EUR{total_annual_savings:,}")
        print(f"Total Investment: EUR{total_investment:,}")
        print(f"Average Payback: {avg_payback:.1f} years")
        print(f"Average ROI: {avg_roi:.1f}%")
        print(f"Total NPV (20 years): EUR{total_npv:,}")
        
        economic_df = pd.DataFrame({
            country_code: {
                **results_by_country[country_code]['economic'],
                'has_real_data': results_by_country[country_code]['has_real_data'],
                'co2_reduction_tons': results_by_country[country_code]['carbon']['annual_co2_reduction_tons']
            } 
            for country_code in results_by_country
        }).T
        
        economic_df.to_csv(f'economic_impact_analysis_{self.country}.csv')
        print(f"\nResults saved to: economic_impact_analysis_{self.country}.csv")
        
        return results_by_country
    
    def generate_comprehensive_report(self, error_reduction_percent, base_improvement=0.05):
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE REPORT FOR {self.country}")
        print(f"{'='*70}")
        
        print(f"\nSCIENTIFIC FINDINGS:")
        print(f"  Forecast Error Reduction: {error_reduction_percent:.1f}% vs persistence")
        
        if error_reduction_percent < 0:
            print(f"  WARNING: Model performs worse than baseline")
        
        carbon_scenarios = self.analyze_carbon_impact_with_uncertainty(error_reduction_percent)
        economic_results = self.analyze_economic_impact(base_improvement, error_reduction_percent)
        
        print(f"\n{'='*70}")
        print("STRATEGIC RECOMMENDATIONS")
        print(f"{'='*70}")
        
        roi_data = []
        for country_code, results in economic_results.items():
            roi = results['economic']['roi_percentage']
            payback = results['economic']['payback_period_years']
            
            if not results['has_real_data']:
                roi *= 0.8
            
            roi_data.append({
                'country': country_code,
                'roi': roi,
                'payback': payback,
                'has_real_data': results['has_real_data']
            })
        
        roi_df = pd.DataFrame(roi_data)
        roi_df = roi_df.sort_values('roi', ascending=False)
        
        print(f"\nPriority Countries:")
        for i, row in roi_df.head(5).iterrows():
            data_indicator = "REAL" if row['has_real_data'] else "EST"
            print(f"  {row['country']}: ROI = {row['roi']:.1f}%, Payback = {row['payback']:.1f} years ({data_indicator})")
        
        report_data = []
        for country_code in economic_results.keys():
            results = economic_results[country_code]
            report_data.append({
                'Country': country_code,
                'Model_Performance_%': error_reduction_percent,
                'ROI_%': results['economic']['roi_percentage'],
                'Payback_Years': results['economic']['payback_period_years'],
                'Annual_Savings_EUR': results['economic']['total_annual_savings_eur'],
                'CO2_Reduction_Tons': results['carbon']['annual_co2_reduction_tons'],
                'Has_Real_Data': results['has_real_data']
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(f'comprehensive_report_{self.country}.csv', index=False)
        
        print(f"\nReport saved to: comprehensive_report_{self.country}.csv")
        
        return report_df
    
    def plot_comprehensive_results(self):
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle(f'Energy Forecasting Results - {self.country}', fontsize=14)
            
            if len(self.training_losses) > 0 and len(self.val_losses) > 0:
                axes[0, 0].plot(self.training_losses, 'b-', label='Training', linewidth=2)
                axes[0, 0].plot(self.val_losses, 'r-', label='Validation', linewidth=2)
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('Training History')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            test_predictions, test_actuals, _ = self.evaluate(self.test_loader, "Test")
            if test_predictions is not None and test_actuals is not None:
                plot_len = min(168, len(test_actuals), len(test_predictions))
                axes[0, 1].plot(test_actuals[:plot_len], 'b-', alpha=0.7, label='Actual', linewidth=1.5)
                axes[0, 1].plot(test_predictions[:plot_len], 'r-', alpha=0.7, label='Predicted', linewidth=1.5)
                axes[0, 1].set_xlabel('Time (hours)')
                axes[0, 1].set_ylabel('Load (MW)')
                axes[0, 1].set_title('Test Set Prediction (1 week)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            if test_predictions is not None and test_actuals is not None:
                min_len = min(len(test_predictions), len(test_actuals))
                errors = test_predictions[:min_len] - test_actuals[:min_len]
                axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black', density=True)
                axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
                axes[1, 0].set_xlabel('Prediction Error (MW)')
                axes[1, 0].set_ylabel('Density')
                axes[1, 0].set_title('Error Distribution')
                axes[1, 0].grid(True, alpha=0.3)
            
            split_colors = ['green', 'orange', 'red']
            split_labels = ['Train (70%)', 'Validation (15%)', 'Test (15%)']
            for i in range(3):
                axes[1, 1].barh(i, 100, color=split_colors[i], alpha=0.5, label=split_labels[i])
            axes[1, 1].set_xlabel('Percentage (%)')
            axes[1, 1].set_title('Data Split')
            axes[1, 1].set_yticks([])
            axes[1, 1].legend()
            
            if hasattr(self, 'y_train_original') and self.y_train_original is not None:
                train_len = min(200, len(self.y_train_original))
                val_len = min(200, len(self.y_val_original))
                test_len = min(200, len(self.y_test_original))
                
                axes[2, 0].plot(self.y_train_original[-train_len:], 'g-', alpha=0.5, label='Train', linewidth=1)
                axes[2, 0].plot(range(train_len, train_len + val_len), 
                               self.y_val_original[:val_len], 'orange', alpha=0.5, label='Val', linewidth=1)
                axes[2, 0].plot(range(train_len + val_len, train_len + val_len + test_len), 
                               self.y_test_original[:test_len], 'r-', alpha=0.5, label='Test', linewidth=1)
                axes[2, 0].set_xlabel('Time Index')
                axes[2, 0].set_ylabel('Load (MW)')
                axes[2, 0].set_title('Load Profile Across Splits')
                axes[2, 0].legend()
                axes[2, 0].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'comprehensive_analysis_results_{self.country}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: comprehensive_analysis_results_{self.country}.png")
            
        except Exception as e:
            print(f"Error creating plot for {self.country}: {e}")
    
    def save_model(self, path=None):
        if self.model is None:
            print("No model to save")
            return
        
        if path is None:
            path = f'scientific_transformer_{self.country}.pth'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'country': self.country,
            'training_losses': self.training_losses,
            'val_losses': self.val_losses
        }, path)
        print(f"Model saved to: {path}")

def get_all_countries(filepath, n_samples=10000):
    df = pd.read_csv(filepath, nrows=n_samples)
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    
    pattern = r'([A-Z]{2})_load_actual_entsoe_transparency'
    countries = []
    for col in df.columns:
        match = re.match(pattern, col)
        if match:
            country_code = match.group(1)
            if country_code not in countries:
                countries.append(country_code)
    
    return sorted(countries)

def main():
    print("=" * 70)
    print("COMPLETE CARBON IMPACT ANALYSIS FOR ALL 31 EUROPEAN COUNTRIES")
    print("=" * 70)
    
    data_path = 'C:/Users/Zahara/Documents/Zoom/europe_energy_forecast/data/europe_energy_real.csv'
    
    all_countries = get_all_countries(data_path)
    print(f"Found {len(all_countries)} countries: {all_countries}")
    
    all_results = {}
    
    for i, country in enumerate(all_countries):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(all_countries)}] Processing: {country}")
        print(f"{'='*70}")
        
        predictor = ScientificEnergyPredictor(sequence_length=168, country=country)
        
        try:
            train_loader, val_loader, test_loader = predictor.load_and_prepare_data(data_path)
            
            if train_loader is None:
                print(f"Skipping {country}: No valid data")
                continue
            
            predictor.build_model()
            predictor.train(epochs=10, lr=0.0005, patience=5)
            
            val_predictions, val_actuals, val_error_reduction = predictor.evaluate(predictor.val_loader, "Validation")
            test_predictions, test_actuals, test_error_reduction = predictor.evaluate(predictor.test_loader, "Test")
            
            base_improvement = 0.05
            comprehensive_report = predictor.generate_comprehensive_report(test_error_reduction, base_improvement)
            
            predictor.plot_comprehensive_results()
            predictor.save_model()
            
            all_results[country] = {
                'val_error_reduction': val_error_reduction,
                'test_error_reduction': test_error_reduction,
                'mae': np.mean(np.abs(test_predictions - test_actuals)) if test_predictions is not None else None
            }
            
            print(f"\n{country} completed successfully")
            
        except Exception as e:
            print(f"Error for {country}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("ALL COUNTRIES PROCESSED")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.sort_values('test_error_reduction', ascending=False)
    results_df.to_csv('all_31_countries_transformer_results.csv')
    
    print("\nCountry Performance Ranking (by Improvement %):")
    print(results_df[['test_error_reduction', 'mae']].to_string())
    
    print(f"\nGenerated Files:")
    print(f"   - carbon_impact_scenarios_[COUNTRY].csv (for each country)")
    print(f"   - economic_impact_analysis_[COUNTRY].csv (for each country)")
    print(f"   - comprehensive_report_[COUNTRY].csv (for each country)")
    print(f"   - scientific_transformer_[COUNTRY].pth (for each country)")
    print(f"   - comprehensive_analysis_results_[COUNTRY].png (for each country)")
    print(f"   - all_31_countries_transformer_results.csv")

if __name__ == "__main__":
    main()
