import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=168):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        return (self.X[idx:idx+self.sequence_length], 
                self.y[idx+self.sequence_length])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ScientificTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        super(ScientificTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True, dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)  # Output mean and log-variance
        )
        
        self._causal_mask = None
    
    def _generate_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, x, return_uncertainty=False):
        batch_size, seq_len, _ = x.shape
        
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        if self._causal_mask is None or self._causal_mask.size(0) != seq_len:
            self._causal_mask = self._generate_causal_mask(seq_len).to(x.device)
        
        x = self.transformer(x, mask=self._causal_mask)
        x = x[:, -1, :]
        
        output = self.output_layer(x)
        mu = output[:, 0]
        log_var = output[:, 1]
        
        if return_uncertainty:
            sigma = torch.exp(0.5 * log_var)
            return mu, sigma
        return mu, log_var

class PublicationReadyTransformer:
    def __init__(self, sequence_length=168, target_country='AT'):
        self.sequence_length = sequence_length
        self.target_country = target_country
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # CHANGE: Use StandardScaler for better uncertainty
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_names = []
        self.lag_feature_indices = []
        self.time_feature_indices = []
        
        print(f"Device: {self.device}")
        print(f"Target Country: {self.target_country}")
    
    def create_scientific_features(self, df, target_col):
        df = df.copy()
        df['target'] = df[target_col]
        
        features_dict = {}
        self.feature_names = []
        
        lags = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168]  # Better lags
        for lag in lags:
            col_name = f'target_lag_{lag}'
            df[col_name] = df['target'].shift(lag)
            df[col_name] = df[col_name].ffill()
            features_dict[col_name] = df[col_name].values
            self.feature_names.append(col_name)
        
        self.lag_feature_indices = list(range(len(lags)))
        
        if 'utc_timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['utc_timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['month'] = df['timestamp'].dt.month
                df['day_of_year'] = df['timestamp'].dt.dayofyear
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(float)
                
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
                df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
                
                time_features = [
                    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                    'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
                    'is_weekend'
                ]
                
                for feat in time_features:
                    features_dict[feat] = df[feat].values
                    self.feature_names.append(feat)
                    
                self.time_feature_indices = list(range(len(lags), len(lags) + len(time_features)))
                    
            except Exception as e:
                print(f"Time features warning: {str(e)[:50]}")
        
        X = np.column_stack([features_dict[name] for name in self.feature_names])
        y = df['target'].values
        
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Created {len(self.feature_names)} features:")
        print(f"  - {len(self.lag_feature_indices)} lag features")
        print(f"  - {len(self.time_feature_indices)} time features")
        
        return X, y
    
    def prepare_publication_ready_data(self, filepath='data/europe_energy_real.csv'):
        print(f"\n{'='*60}")
        print("DATA PREPARATION (Publication Ready - Three-way split)")
        print(f"{'='*60}")
        
        df = pd.read_csv(filepath, nrows=30000)
        
        target_col = f'{self.target_country}_load_actual_entsoe_transparency'
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        X, y = self.create_scientific_features(df, target_col)
        
        train_idx = int(len(X) * 0.7)
        val_idx = int(len(X) * 0.85)
        
        X_train_raw = X[:train_idx]
        X_val_raw = X[train_idx:val_idx]
        X_test_raw = X[val_idx:]
        
        y_train_raw = y[:train_idx]
        y_val_raw = y[train_idx:val_idx]
        y_test_raw = y[val_idx:]
        
        print(f"\nThree-way time-series split:")
        print(f"  Training: {len(X_train_raw):,} samples ({len(X_train_raw)/24:.1f} days)")
        print(f"  Validation: {len(X_val_raw):,} samples ({len(X_val_raw)/24:.1f} days)")
        print(f"  Testing: {len(X_test_raw):,} samples ({len(X_test_raw)/24:.1f} days)")
        
        X_train = self.feature_scaler.fit_transform(X_train_raw)
        X_val = self.feature_scaler.transform(X_val_raw)
        X_test = self.feature_scaler.transform(X_test_raw)
        
        y_train = self.target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        y_val = self.target_scaler.transform(y_val_raw.reshape(-1, 1)).flatten()
        y_test = self.target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
        
        self.X_train_raw = X_train_raw
        self.X_val_raw = X_val_raw
        self.X_test_raw = X_test_raw
        self.y_train_raw = y_train_raw
        self.y_val_raw = y_val_raw
        self.y_test_raw = y_test_raw
        
        train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
        test_dataset = TimeSeriesDataset(X_test, y_test, self.sequence_length)
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Smaller batch
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.input_dim = X_train.shape[1]
        
        print(f"\nFinal configuration:")
        print(f"  Input dimension: {self.input_dim}")
        print(f"  Sequence length: {self.sequence_length} hours")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def build_model(self):
        self.model = ScientificTransformer(
            input_dim=self.input_dim,
            d_model=64,  # Increased
            nhead=8,     # Increased
            num_layers=3, # Increased
            dropout=0.2   # Increased
        ).to(self.device)
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel built: {params:,} parameters")
        print(f"Features: Positional Encoding, Causal Mask, End-to-end Uncertainty")
        return self.model
    
    def gaussian_nll_loss(self, mu, log_var, target, eps=1e-6):
        """Gaussian NLL with numerical stability"""
        var = torch.exp(log_var) + eps
        return 0.5 * torch.mean(log_var + (target - mu)**2 / var)
    
    def crps_gaussian(self, mu, sigma, target, eps=1e-6):
        """CRPS for Gaussian with numerical stability"""
        sigma = sigma + eps
        z = (target - mu) / sigma
        cdf = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
        pdf = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
        return torch.mean(crps)
    
    def train_publication_ready(self, epochs=30, lr=0.0005):  # More epochs, lower LR
        if self.model is None:
            self.build_model()
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        print(f"\n{'='*60}")
        print("TRAINING (Improved End-to-end Uncertainty Learning)")
        print(f"{'='*60}")
        
        train_nll_losses = []
        val_nll_losses = []
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_nll = 0
            train_mse = 0
            
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                optimizer.zero_grad()
                
                mu, log_var = self.model(X_batch)
                
                # Combined loss with uncertainty regularization
                nll_loss = self.gaussian_nll_loss(mu, log_var, y_batch)
                mse_loss = torch.mean((mu - y_batch)**2)
                
                # Regularize uncertainty to prevent collapse
                sigma = torch.exp(0.5 * log_var)
                uncertainty_reg = torch.mean(torch.abs(1.0 - sigma))
                
                combined_loss = nll_loss + 0.1 * mse_loss + 0.01 * uncertainty_reg
                
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_nll += nll_loss.item()
                train_mse += mse_loss.item()
            
            avg_train_nll = train_nll / len(self.train_loader)
            avg_train_mse = train_mse / len(self.train_loader)
            
            self.model.eval()
            val_nll = 0
            val_mse = 0
            
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch = X_batch.float().to(self.device)
                    y_batch = y_batch.float().to(self.device)
                    
                    mu, log_var = self.model(X_batch)
                    
                    nll_loss = self.gaussian_nll_loss(mu, log_var, y_batch)
                    mse_loss = torch.mean((mu - y_batch)**2)
                    
                    val_nll += nll_loss.item()
                    val_mse += mse_loss.item()
            
            avg_val_nll = val_nll / len(self.val_loader)
            avg_val_mse = val_mse / len(self.val_loader)
            
            train_nll_losses.append(avg_train_nll)
            val_nll_losses.append(avg_val_nll)
            
            scheduler.step(avg_val_nll)
            
            if avg_val_nll < best_val_loss:
                best_val_loss = avg_val_nll
                best_model_state = self.model.state_dict().copy()
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d} | LR: {current_lr:.6f} | "
                      f"Train NLL: {avg_train_nll:.6f} MSE: {avg_train_mse:.6f} | "
                      f"Val NLL: {avg_val_nll:.6f} MSE: {avg_val_mse:.6f}")
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nLoaded best model with val NLL: {best_val_loss:.6f}")
        
        print(f"\nTraining completed.")
        print(f"Final validation NLL: {val_nll_losses[-1]:.6f}")
        
        return train_nll_losses, val_nll_losses
    
    def evaluate_with_uncertainty(self, loader):
        self.model.eval()
        
        predictions = []
        actuals = []
        uncertainties = []
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                mu, log_var = self.model(X_batch)
                sigma = torch.exp(0.5 * log_var)
                
                predictions.extend(mu.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())
                uncertainties.extend(sigma.cpu().numpy())
        
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        uncertainties_array = np.array(uncertainties)
        
        predictions_original = self.target_scaler.inverse_transform(
            predictions_array.reshape(-1, 1)
        ).flatten()
        
        actuals_original = self.target_scaler.inverse_transform(
            actuals_array.reshape(-1, 1)
        ).flatten()
        
        # For StandardScaler, uncertainty scales with std
        uncertainties_original = uncertainties_array * self.target_scaler.scale_[0]
        
        return predictions_original, actuals_original, uncertainties_original
    
    def calculate_prediction_intervals(self, predictions, uncertainties, confidence=0.95):
        z_score = 1.96
        lower_bound = predictions - z_score * uncertainties
        upper_bound = predictions + z_score * uncertainties
        return lower_bound, upper_bound
    
    def evaluate_publication_ready(self):
        if self.model is None:
            print("Model not trained")
            return None, None, None, None, None, None, None
        
        print(f"\n{'='*80}")
        print("FINAL TESTING ON HELD-OUT TEST SET")
        print(f"{'='*80}")
        
        predictions, actuals, uncertainties = self.evaluate_with_uncertainty(self.test_loader)
        lower_bound, upper_bound = self.calculate_prediction_intervals(predictions, uncertainties)
        
        val_predictions, val_actuals, val_uncertainties = self.evaluate_with_uncertainty(self.val_loader)
        
        lag1_idx = self.feature_names.index('target_lag_1')
        lag24_idx = self.feature_names.index('target_lag_24')
        lag168_idx = self.feature_names.index('target_lag_168')
        
        lag_1_test = self.X_test_raw[self.sequence_length:, lag1_idx]
        lag_24_test = self.X_test_raw[self.sequence_length:, lag24_idx]
        lag_168_test = self.X_test_raw[self.sequence_length:, lag168_idx]
        
        mean_baseline = np.full_like(actuals, self.y_train_raw.mean())
        
        min_len = min(len(predictions), len(actuals), 
                     len(lag_1_test), len(lag_168_test))
        
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]
        uncertainties = uncertainties[:min_len]
        lower_bound = lower_bound[:min_len]
        upper_bound = upper_bound[:min_len]
        persistence_baseline = lag_1_test[:min_len]
        daily_persistence = lag_24_test[:min_len]
        weekly_baseline = lag_168_test[:min_len]
        mean_baseline = mean_baseline[:min_len]
        
        normalized_errors = (actuals - predictions) / (uncertainties + 1e-6)
        
        metrics = {}
        
        coverage = np.mean((actuals >= lower_bound) & (actuals <= upper_bound))
        avg_width = np.mean(upper_bound - lower_bound)
        
        metrics['transformer'] = {
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mae': mean_absolute_error(actuals, predictions),
            'r2': r2_score(actuals, predictions),
            'uncertainty_metrics': {
                'coverage_95': coverage * 100,
                'avg_interval_width': avg_width,
                'avg_uncertainty': np.mean(uncertainties),
            }
        }
        
        metrics['persistence'] = {
            'rmse': np.sqrt(mean_squared_error(actuals, persistence_baseline)),
            'mae': mean_absolute_error(actuals, persistence_baseline),
            'r2': r2_score(actuals, persistence_baseline)
        }
        
        metrics['daily_persistence'] = {
            'rmse': np.sqrt(mean_squared_error(actuals, daily_persistence)),
            'mae': mean_absolute_error(actuals, daily_persistence),
            'r2': r2_score(actuals, daily_persistence)
        }
        
        metrics['weekly_persistence'] = {
            'rmse': np.sqrt(mean_squared_error(actuals, weekly_baseline)),
            'mae': mean_absolute_error(actuals, weekly_baseline),
            'r2': r2_score(actuals, weekly_baseline)
        }
        
        metrics['mean_baseline'] = {
            'rmse': np.sqrt(mean_squared_error(actuals, mean_baseline)),
            'mae': mean_absolute_error(actuals, mean_baseline),
            'r2': r2_score(actuals, mean_baseline)
        }
        
        for baseline_name in ['persistence', 'daily_persistence', 'weekly_persistence', 'mean_baseline']:
            baseline_rmse = metrics[baseline_name]['rmse']
            model_rmse = metrics['transformer']['rmse']
            improvement = ((baseline_rmse - model_rmse) / baseline_rmse) * 100
            metrics[baseline_name]['improvement'] = improvement
        
        print(f"\nTransformer Model Performance (TEST SET):")
        print(f"  RMSE: {metrics['transformer']['rmse']:.2f} MW")
        print(f"  MAE:  {metrics['transformer']['mae']:.2f} MW")
        print(f"  R²:   {metrics['transformer']['r2']:.4f}")
        
        print(f"\nUncertainty Estimation:")
        print(f"  95% Coverage: {metrics['transformer']['uncertainty_metrics']['coverage_95']:.1f}%")
        print(f"  Average Interval Width: {metrics['transformer']['uncertainty_metrics']['avg_interval_width']:.1f} MW")
        print(f"  Average Uncertainty: {metrics['transformer']['uncertainty_metrics']['avg_uncertainty']:.1f} MW")
        
        print(f"\nBaseline Comparisons:")
        print(f"  1. Persistence (1-hour):")
        print(f"     RMSE: {metrics['persistence']['rmse']:.2f} MW")
        print(f"     Improvement: {metrics['persistence']['improvement']:.1f}%")
        
        print(f"  2. Daily Persistence (24-hour):")
        print(f"     RMSE: {metrics['daily_persistence']['rmse']:.2f} MW")
        print(f"     Improvement: {metrics['daily_persistence']['improvement']:.1f}%")
        
        print(f"  3. Weekly Persistence (168-hour):")
        print(f"     RMSE: {metrics['weekly_persistence']['rmse']:.2f} MW")
        print(f"     Improvement: {metrics['weekly_persistence']['improvement']:.1f}%")
        
        print(f"  4. Mean Baseline:")
        print(f"     RMSE: {metrics['mean_baseline']['rmse']:.2f} MW")
        print(f"     Improvement: {metrics['mean_baseline']['improvement']:.1f}%")
        
        return predictions, actuals, metrics, uncertainties, normalized_errors, val_predictions, val_actuals
    
    def forecast_publication_ready(self, n_days=7, start_date=None):
        if self.model is None:
            print("Model not trained")
            return
        
        self.model.eval()
        print(f"\n{'='*60}")
        print(f"PUBLICATION-READY FORECASTING WITH UNCERTAINTY ({n_days} days)")
        print(f"{'='*60}")
        
        # Use the most recent data for initialization
        last_sequence_scaled = self.feature_scaler.transform(
            self.X_test_raw[-self.sequence_length:]
        )
        last_sequence_raw = self.X_test_raw[-self.sequence_length:].copy()
        
        if start_date is None:
            start_time = pd.Timestamp.now().replace(minute=0, second=0, microsecond=0)
        else:
            start_time = pd.Timestamp(start_date)
        
        forecasts = []
        uncertainties = []
        forecast_times = []
        lower_bounds = []
        upper_bounds = []
        
        current_features_raw = last_sequence_raw.copy()
        current_features_scaled = last_sequence_scaled.copy()
        
        with torch.no_grad():
            current_time = start_time
            
            for hour in range(n_days * 24):
                input_tensor = torch.FloatTensor(current_features_scaled).unsqueeze(0).to(self.device)
                mu, log_var = self.model(input_tensor)
                sigma = torch.exp(0.5 * log_var)
                
                pred_scaled = mu.cpu().numpy().item()
                uncertainty_scaled = sigma.cpu().numpy().item()
                
                pred_original = self.target_scaler.inverse_transform(
                    np.array([[pred_scaled]])
                ).item()
                
                uncertainty_original = uncertainty_scaled * self.target_scaler.scale_[0]
                
                forecasts.append(pred_original)
                uncertainties.append(uncertainty_original)
                forecast_times.append(current_time)
                
                # Calculate prediction intervals
                lower = pred_original - 1.96 * uncertainty_original
                upper = pred_original + 1.96 * uncertainty_original
                lower_bounds.append(lower)
                upper_bounds.append(upper)
                
                # Update features for next time step
                new_features_raw = current_features_raw[-1].copy()
                
                # Shift lag features
                for lag_idx in reversed(self.lag_feature_indices[1:]):
                    new_features_raw[lag_idx] = new_features_raw[lag_idx - 1]
                
                # Set new prediction as lag_1
                new_features_raw[self.lag_feature_indices[0]] = pred_original
                
                # Update time features
                next_time = current_time + timedelta(hours=1)
                hour_val = next_time.hour
                day_of_week = next_time.dayofweek
                month = next_time.month
                day_of_year = next_time.dayofyear
                
                # Update all time features
                time_idx = 0
                for i in range(len(self.lag_feature_indices), len(self.feature_names)):
                    feat_name = self.feature_names[i]
                    if 'hour_sin' in feat_name:
                        new_features_raw[i] = np.sin(2 * np.pi * hour_val / 24)
                    elif 'hour_cos' in feat_name:
                        new_features_raw[i] = np.cos(2 * np.pi * hour_val / 24)
                    elif 'day_sin' in feat_name:
                        new_features_raw[i] = np.sin(2 * np.pi * day_of_week / 7)
                    elif 'day_cos' in feat_name:
                        new_features_raw[i] = np.cos(2 * np.pi * day_of_week / 7)
                    elif 'month_sin' in feat_name:
                        new_features_raw[i] = np.sin(2 * np.pi * month / 12)
                    elif 'month_cos' in feat_name:
                        new_features_raw[i] = np.cos(2 * np.pi * month / 12)
                    elif 'doy_sin' in feat_name:
                        new_features_raw[i] = np.sin(2 * np.pi * day_of_year / 365.25)
                    elif 'doy_cos' in feat_name:
                        new_features_raw[i] = np.cos(2 * np.pi * day_of_year / 365.25)
                    elif 'is_weekend' in feat_name:
                        new_features_raw[i] = 1.0 if day_of_week in [5, 6] else 0.0
                
                # Scale and update sequences
                new_features_scaled = self.feature_scaler.transform(
                    new_features_raw.reshape(1, -1)
                )[0]
                
                current_features_raw = np.roll(current_features_raw, -1, axis=0)
                current_features_raw[-1] = new_features_raw
                
                current_features_scaled = np.roll(current_features_scaled, -1, axis=0)
                current_features_scaled[-1] = new_features_scaled
                
                current_time = next_time
                
                if hour % 24 == 0 and hour > 0:
                    print(f"  Day {hour//24}/{n_days} - Current time: {current_time}")
        
        # Create forecast dataframe - FIXED: Ensure all arrays are 1D
        forecast_df = pd.DataFrame({
            'timestamp': forecast_times,
            'predicted_load_MW': forecasts,
            'uncertainty_MW': uncertainties,
            'lower_bound_95': lower_bounds,
            'upper_bound_95': upper_bounds,
            'hour_of_day': [t.hour for t in forecast_times],
            'day_of_week': [t.dayofweek for t in forecast_times]
        })
        
        output_path = f'publication_forecast_{self.target_country}_{n_days}days.csv'
        forecast_df.to_csv(output_path, index=False)
        
        print(f"\nForecast saved: {output_path}")
        print(f"Forecast period: {start_time.date()} to {current_time.date()}")
        print(f"Average forecast: {np.mean(forecasts):.1f} ± {np.mean(uncertainties):.1f} MW")
        print(f"Peak forecast: {np.max(forecasts):.1f} MW")
        print(f"Average uncertainty: {np.mean(uncertainties):.1f} MW")
        
        return forecast_df

def main():
    print("=" * 80)
    print("IMPROVED PUBLICATION-READY TRANSFORMER FORECASTING")
    print("=" * 80)
    
    forecaster = PublicationReadyTransformer(
        sequence_length=168,
        target_country='AT'
    )
    
    try:
        print("\nPHASE 1: Data Preparation (Three-way split)")
        print("-" * 40)
        train_loader, val_loader, test_loader = forecaster.prepare_publication_ready_data()
        
        print("\nPHASE 2: Model Building")
        print("-" * 40)
        forecaster.build_model()
        
        print("\nPHASE 3: Improved Model Training")
        print("-" * 40)
        train_nll, val_nll = forecaster.train_publication_ready(epochs=30)
        
        forecaster.train_history_nll = train_nll
        forecaster.val_history_nll = val_nll
        
        print("\nPHASE 4: Comprehensive Testing")
        print("-" * 40)
        results = forecaster.evaluate_publication_ready()
        
        if results[0] is not None:
            predictions, actuals, metrics, uncertainties, normalized_errors, val_predictions, val_actuals = results
            
            print("\nPHASE 5: Publication-Ready Forecasting")
            print("-" * 40)
            forecast = forecaster.forecast_publication_ready(n_days=7)
            
            print(f"\n{'='*80}")
            print("PROCESS COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            
            print("\nGenerated Files:")
            print(f"  1. publication_forecast_{forecaster.target_country}_7days.csv")
            
            print("\nKey Improvements:")
            print("✓ Larger model architecture")
            print("✓ StandardScaler for better uncertainty calibration")
            print("✓ Better lag features")
            print("✓ Learning rate scheduling")
            print("✓ Uncertainty regularization")
            print("✓ Fixed forecasting bug")
            
        else:
            print("Evaluation failed")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
