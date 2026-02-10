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
warnings.filterwarnings('ignore')

# ============================================================================
# ANALYZER CLASSES - FINAL FIX
# ============================================================================

class CarbonImpactAnalyzer:
    def __init__(self):
        self.co2_intensity_by_country = {
            'DE': 420, 'FR': 56, 'SE': 40, 'AT': 120, 'ES': 230,
            'IT': 320, 'GB': 250, 'NL': 390, 'PL': 710, 'BE': 180,
            'DK': 150, 'FI': 120, 'IE': 350, 'PT': 260, 'GR': 580,
            'CZ': 530, 'HU': 280, 'RO': 340, 'BG': 490, 'HR': 280,
            'SI': 280, 'SK': 220, 'EE': 560, 'LV': 160, 'LT': 120,
            'LU': 200, 'MT': 480, 'CY': 650
        }
    
    def calculate_carbon_reduction(self, df, improvement, country_code='AT', use_default_only=False):
        # FIX: If improvement is 0 or negative, return ZERO
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
            if use_default_only:
                result = self._get_default_values(country_code)
                # Scale default values by improvement
                result['annual_co2_reduction_tons'] *= improvement / 0.05  # Scale from 5% base
                result['annual_energy_savings_mwh'] *= improvement / 0.05
                result['equivalent_cars_removed'] = int(result['annual_co2_reduction_tons'] / 4.6)
                result['equivalent_trees_planted'] = int(result['annual_co2_reduction_tons'] * 20)
                return result
            
            load_col = f"{country_code.lower()}_load_actual_entsoe_transparency"
            
            if load_col not in df.columns:
                possible_patterns = [
                    f"{country_code.lower()}_load_actual",
                    f"load_actual_{country_code.lower()}",
                    f"{country_code.lower()}_load",
                ]
                
                found_col = None
                for pattern in possible_patterns:
                    matching_cols = [col for col in df.columns if pattern in col.lower()]
                    if matching_cols:
                        found_col = matching_cols[0]
                        break
                
                if found_col:
                    load_col = found_col
                else:
                    result = self._get_default_values(country_code)
                    result['annual_co2_reduction_tons'] *= improvement / 0.05
                    result['annual_energy_savings_mwh'] *= improvement / 0.05
                    result['equivalent_cars_removed'] = int(result['annual_co2_reduction_tons'] / 4.6)
                    result['equivalent_trees_planted'] = int(result['annual_co2_reduction_tons'] * 20)
                    return result
            
            if load_col not in df.columns or df[load_col].isna().all():
                result = self._get_default_values(country_code)
                result['annual_co2_reduction_tons'] *= improvement / 0.05
                result['annual_energy_savings_mwh'] *= improvement / 0.05
                result['equivalent_cars_removed'] = int(result['annual_co2_reduction_tons'] / 4.6)
                result['equivalent_trees_planted'] = int(result['annual_co2_reduction_tons'] * 20)
                return result
            
            avg_consumption = df[load_col].mean()
            
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
                    hours_per_year = 8760 if abs(time_diff.total_seconds() / 3600 - 1) < 0.1 else 365*24
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
        base_co2 = 50000 * multiplier  # Base at 5% improvement
        
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
        # FIX: If improvement is 0 or negative, return ZERO economic impact
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
            # Apply model performance adjustment
            if model_performance < 0:
                adjusted_improvement = improvement * max(0, 1 + model_performance/100)
                improvement = adjusted_improvement
            
            # FIX: Double-check after adjustment
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
                co2_reduction = 50000 * (improvement / 0.05)  # Scale from 5% base
            
            price_cols = [col for col in df.columns if 'price' in col.lower() and country_code.lower() in col.lower()]
            
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
                # Calculate based on improvement
                energy_savings_mwh = 1000000 * (improvement / 0.05)  # Scale from 5% base
            
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

# Keep the rest of the code EXACTLY THE SAME as before
# (Transformer, PositionalEncoding, TimeSeriesDataset, ScientificEnergyPredictor classes)
# Only replace the CarbonImpactAnalyzer and EconomicAnalyzer classes above

# ============================================================================
# SCIENTIFIC TRANSFORMER MODEL (EXACTLY SAME)
# ============================================================================

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

# ============================================================================
# SCIENTIFIC ENERGY PREDICTOR (EXACTLY SAME)
# ============================================================================

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
    
    # ALL OTHER METHODS REMAIN EXACTLY THE SAME AS IN PREVIOUS CODE
    # Only the analyzers are updated
    
    def prepare_features_scientifically(self, df, target_col):
        features = []
        
        for lag in [1, 2, 3, 4, 5, 6, 7, 24, 48, 72, 168]:
            if len(df) > lag:
                features.append(df[target_col].shift(lag).values.reshape(-1, 1))
        
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
        features = pd.DataFrame(features).fillna(method='bfill').fillna(method='ffill').values
        
        return features
    
    def load_and_prepare_data(self, filepath='data/europe_energy_real.csv'):
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        
        if 'utc_timestamp' in df.columns:
            df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
            df.set_index('utc_timestamp', inplace=True)
        
        target_col = f"{self.country}_load_actual_entsoe_transparency"
        
        if target_col not in df.columns:
            possible_targets = [
                f"{self.country.lower()}_load_actual",
                f"load_actual_{self.country.lower()}",
                f"{self.country.lower()}_load",
            ]
            
            for possible in possible_targets:
                matching = [col for col in df.columns if possible in col.lower()]
                if matching:
                    target_col = matching[0]
                    break
            
            if target_col == f"{self.country}_load_actual_entsoe_transparency":
                raise ValueError(f"Target column for {self.country} not found")
        
        X = self.prepare_features_scientifically(df, target_col)
        y = df[target_col].values
        
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
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
        
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        persistence_forecast = np.roll(actuals, 24)
        valid_idx = ~np.isnan(persistence_forecast)
        persistence_mae = np.mean(np.abs(actuals[valid_idx] - persistence_forecast[valid_idx]))
        
        error_reduction = (persistence_mae - mae) / persistence_mae * 100
        
        print(f"\n{'='*60}")
        print(f"EVALUATION ON {set_name.upper()} SET")
        print(f"{'='*60}")
        print(f"RMSE: {rmse:.2f} MW")
        print(f"MAE: {mae:.2f} MW")
        print(f"24h Persistence MAE: {persistence_mae:.2f} MW")
        print(f"Improvement over persistence: {error_reduction:.1f}%")
        print(f"Predictions range: [{predictions.min():.1f}, {predictions.max():.1f}] MW")
        print(f"Actuals range: [{actuals.min():.1f}, {actuals.max():.1f}] MW")
        
        return predictions, actuals, error_reduction
    
    def forecast_future_scientifically(self, n_days=3, use_teacher_forcing=True):
        if self.model is None:
            print("Model not trained")
            return None
        
        self.model.eval()
        
        print(f"\nGenerating {n_days}-day forecast...")
        print(f"Mode: {'Teacher-forcing' if use_teacher_forcing else 'Free-running'}")
        
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
        
        forecast_csv_path = f'scientific_forecast_{n_days}days.csv'
        forecast_df.to_csv(forecast_csv_path, index=False)
        
        print(f"\nForecast saved to: {forecast_csv_path}")
        print(f"Forecast range: [{np.min(forecasts):.1f}, {np.max(forecasts):.1f}] MW")
        print(f"Average: {np.mean(forecasts):.1f} MW")
        
        return forecast_df
    
    def _check_country_data_availability(self, country_code):
        patterns = [
            f"{country_code.lower()}_load_actual",
            f"load_actual_{country_code.lower()}",
            f"{country_code.lower()}_load",
        ]
        
        for pattern in patterns:
            matching = [col for col in self.original_data.columns if pattern in col.lower()]
            if matching:
                col = matching[0]
                if col in self.original_data.columns:
                    data = self.original_data[col]
                    if not data.isna().all() and data.mean() > 0:
                        return True
        return False
    
    def analyze_carbon_impact_with_uncertainty(self, error_reduction_percent, 
                                              improvement_scenarios=[0.01, 0.05, 0.10]):
        print(f"\n{'='*60}")
        print("CARBON IMPACT ANALYSIS WITH UNCERTAINTY")
        print(f"{'='*60}")
        
        if self.original_data is None:
            print("No data available")
            return {}
        
        print(f"\nModel Performance: {error_reduction_percent:.1f}% vs persistence")
        
        # ULTRA-CONSERVATIVE ADJUSTMENT
        if error_reduction_percent < 0:
            print(f"\nWARNING: Model performs WORSE than baseline")
            print(f"Applying ultra-conservative adjustment...")
            
            adjusted_scenarios = []
            for scenario in improvement_scenarios:
                # If model worse than -10%, NO improvement
                if error_reduction_percent < -10:
                    adjusted = 0
                else:
                    # Linear reduction based on performance
                    adjusted = scenario * max(0, 1 + error_reduction_percent/100)
                adjusted_scenarios.append(adjusted)
            
            print(f"Original scenarios: {[f'{s*100:.1f}%' for s in improvement_scenarios]}")
            print(f"Adjusted scenarios: {[f'{s*100:.1f}%' for s in adjusted_scenarios]}")
            improvement_scenarios = adjusted_scenarios
        
        scenario_results = {}
        country_codes = ['AT', 'DE', 'FR', 'IT', 'ES', 'GB', 'NL', 'PL', 'BE']
        
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
            
            scenario_results[f'scenario_{improvement}'] = {
                'improvement_percent': improvement * 100,
                'total_co2_reduction_tons': scenario_co2,
                'total_energy_savings_mwh': scenario_energy,
                'model_performance': error_reduction_percent
            }
            
            print(f"Total CO2 Reduction: {scenario_co2:,.0f} tons")
            print(f"Total Energy Savings: {scenario_energy:,.0f} MWh")
        
        min_co2 = scenario_results[f'scenario_{improvement_scenarios[0]}']['total_co2_reduction_tons']
        max_co2 = scenario_results[f'scenario_{improvement_scenarios[-1]}']['total_co2_reduction_tons']
        
        print(f"\n{'='*60}")
        print("SCENARIO SUMMARY")
        print(f"{'='*60}")
        
        if error_reduction_percent < -10:
            print("ULTRA-CONSERVATIVE: Model performs significantly worse")
            print("Results represent UPPER BOUND estimates")
        elif error_reduction_percent < 0:
            print("CONSERVATIVE: Model performs worse than baseline")
            print("Results adjusted downward")
        
        for i, scenario in enumerate(improvement_scenarios):
            result = scenario_results[f'scenario_{scenario}']
            print(f"{scenario*100:.1f}% improvement: {result['total_co2_reduction_tons']:,.0f} tons CO2/year")
        
        print(f"Range: {min_co2:,.0f} - {max_co2:,.0f} tons CO2/year")
        
        scenario_df = pd.DataFrame(scenario_results).T
        scenario_df.to_csv('carbon_impact_scenarios.csv')
        
        print(f"\nResults saved to: carbon_impact_scenarios.csv")
        
        return scenario_results
    
    def analyze_economic_impact(self, improvement=0.05, model_performance=0.0):
        print(f"\n{'='*60}")
        print("ECONOMIC IMPACT ANALYSIS")
        print(f"{'='*60}")
        
        if self.original_data is None:
            print("No data available")
            return {}
        
        print(f"\nAssumptions:")
        print(f"  Base Efficiency Improvement: {improvement*100:.1f}%")
        print(f"  Model Performance vs Persistence: {model_performance:.1f}%")
        print(f"  Carbon Price: €{self.economic_analyzer.carbon_price}/ton")
        print(f"  Discount Rate: {self.economic_analyzer.discount_rate*100:.1f}%")
        print(f"  Investment Cost: €500/MWh of savings")
        print(f"  Analysis Period: 20 years")
        
        if model_performance < 0:
            print(f"\nMODEL PERFORMANCE WARNING:")
            print(f"  Model performs {model_performance:.1f}% worse than baseline")
            print(f"  Efficiency gains will be reduced accordingly")
        
        results_by_country = {}
        country_codes = ['AT', 'DE', 'FR', 'IT', 'ES', 'GB', 'NL', 'PL', 'BE']
        
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
            print(f"  Annual Savings: €{economic_result['total_annual_savings_eur']:,}")
            print(f"  Investment: €{economic_result['initial_investment_eur']:,}")
            print(f"  Payback: {economic_result['payback_period_years']} years")
            print(f"  ROI: {economic_result['roi_percentage']:.1f}%")
            print(f"  NPV: €{economic_result['npv_eur']:,}")
            
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
        
        print(f"Total Annual Savings: €{total_annual_savings:,}")
        print(f"Total Investment: €{total_investment:,}")
        print(f"Average Payback: {avg_payback:.1f} years")
        print(f"Average ROI: {avg_roi:.1f}%")
        print(f"Total NPV (20 years): €{total_npv:,}")
        
        if model_performance < -10:
            print(f"\nRISK WARNING:")
            print(f"  Model performance is significantly worse than baseline")
            print(f"  These results are UPPER BOUND estimates")
        
        economic_df = pd.DataFrame({
            country_code: {
                **results_by_country[country_code]['economic'],
                'has_real_data': results_by_country[country_code]['has_real_data'],
                'co2_reduction_tons': results_by_country[country_code]['carbon']['annual_co2_reduction_tons'],
                'energy_savings_mwh': results_by_country[country_code]['carbon']['annual_energy_savings_mwh']
            } 
            for country_code in results_by_country
        }).T
        
        economic_df.to_csv('economic_impact_analysis.csv')
        
        print(f"\nResults saved to: economic_impact_analysis.csv")
        
        return results_by_country
    
    def generate_comprehensive_report(self, error_reduction_percent, 
                                     base_improvement=0.05):
        print(f"\n{'='*70}")
        print("COMPREHENSIVE SCIENTIFIC & BUSINESS REPORT")
        print(f"{'='*70}")
        
        print(f"\nSCIENTIFIC FINDINGS:")
        print(f"  Forecast Error Reduction: {error_reduction_percent:.1f}% vs persistence")
        
        if error_reduction_percent < 0:
            print(f"  WARNING: Model performs WORSE than baseline")
            print(f"  Efficiency gains reduced accordingly")
        
        carbon_scenarios = self.analyze_carbon_impact_with_uncertainty(
            error_reduction_percent
        )
        
        economic_results = self.analyze_economic_impact(
            base_improvement, error_reduction_percent
        )
        
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
                'npv': results['economic']['npv_eur'],
                'has_real_data': results['has_real_data']
            })
        
        roi_df = pd.DataFrame(roi_data)
        roi_df = roi_df.sort_values('roi', ascending=False)
        
        print(f"\n1. Priority Countries:")
        for i, row in roi_df.head(3).iterrows():
            data_indicator = "REAL" if row['has_real_data'] else "EST"
            print(f"   {row['country']}: ROI = {row['roi']:.1f}%, "
                  f"Payback = {row['payback']:.1f} years ({data_indicator})")
        
        total_investment = sum(r['economic']['initial_investment_eur'] 
                              for r in economic_results.values())
        total_savings = sum(r['economic']['total_annual_savings_eur'] 
                           for r in economic_results.values())
        
        if error_reduction_percent < -10:
            investment_multiplier = 0.5
            print(f"\n2. Investment Strategy (HIGH RISK):")
            print(f"   Model performs significantly worse than baseline")
            print(f"   Recommended: Pilot projects only")
        elif error_reduction_percent < 0:
            investment_multiplier = 0.7
            print(f"\n2. Investment Strategy (MEDIUM RISK):")
            print(f"   Model performs worse than baseline")
            print(f"   Recommended: Conservative phased investment")
        else:
            investment_multiplier = 1.0
            print(f"\n2. Investment Strategy (LOW RISK):")
            print(f"   Model performs better than baseline")
            print(f"   Recommended: Full implementation")
        
        phase1_investment = total_investment * investment_multiplier * 0.5
        print(f"   Phase 1: €{phase1_investment:,.0f}")
        print(f"   Full Implementation: €{total_investment:,.0f}")
        print(f"   Annual Savings: €{total_savings:,.0f}")
        
        report_data = []
        for country_code in economic_results.keys():
            results = economic_results[country_code]
            report_data.append({
                'Country': country_code,
                'Model_Performance_%': error_reduction_percent,
                'Base_Improvement_%': base_improvement * 100,
                'Adjusted_Improvement_%': results['economic']['adjusted_improvement_percent'],
                'Annual_Savings_EUR': results['economic']['total_annual_savings_eur'],
                'Investment_EUR': results['economic']['initial_investment_eur'],
                'ROI_%': results['economic']['roi_percentage'],
                'Payback_Years': results['economic']['payback_period_years'],
                'NPV_EUR': results['economic']['npv_eur'],
                'CO2_Reduction_Tons': results['carbon']['annual_co2_reduction_tons'],
                'Energy_Savings_MWh': results['carbon']['annual_energy_savings_mwh'],
                'Has_Real_Data': results['has_real_data'],
                'Risk_Assessment': 'HIGH' if error_reduction_percent < -10 else 'MEDIUM' if error_reduction_percent < 0 else 'LOW'
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv('comprehensive_scientific_report.csv', index=False)
        
        print(f"\nReport saved to: comprehensive_scientific_report.csv")
        
        if error_reduction_percent < 0:
            print(f"\nFINAL RECOMMENDATION:")
            print(f"  Focus on model improvement before large investments")
        else:
            print(f"\nFINAL RECOMMENDATION:")
            print(f"  Proceed with investment strategy")
        
        return report_df
    
    def plot_comprehensive_results(self):
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            
            if hasattr(self, 'training_losses') and hasattr(self, 'val_losses'):
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
                axes[0, 1].set_title('Test Set: One Week Prediction')
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
            
            if hasattr(self, 'y_train_original'):
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
            
            if test_predictions is not None and test_actuals is not None:
                baseline_labels = ['Our Model', '24h Persistence']
                mae_values = []
                
                min_len = min(len(test_predictions), len(test_actuals))
                if min_len > 0:
                    mae_our = np.mean(np.abs(test_predictions[:min_len] - test_actuals[:min_len]))
                    mae_values.append(mae_our)
                    
                    if min_len > 24:
                        persistence_actuals = test_actuals[:min_len-24]
                        persistence_forecast = test_actuals[24:min_len]
                        mae_persistence = np.mean(np.abs(persistence_actuals - persistence_forecast))
                        mae_values.append(mae_persistence)
                
                if mae_values:
                    axes[2, 1].bar(baseline_labels[:len(mae_values)], mae_values, alpha=0.7)
                    axes[2, 1].set_ylabel('MAE (MW)')
                    axes[2, 1].set_title('Comparison with Baseline')
                    axes[2, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('comprehensive_analysis_results.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Comprehensive plot saved to: comprehensive_analysis_results.png")
            
        except Exception as e:
            print(f"Error creating plot: {e}")
    
    def save_model(self, path='scientific_business_transformer.pth'):
        if self.model is None:
            print("No model to save")
            return
        
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

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("SCIENTIFIC ENERGY FORECASTING WITH BUSINESS ANALYSIS")
    print("=" * 70)
    
    predictor = ScientificEnergyPredictor(sequence_length=168, country='AT')
    
    try:
        print(f"\n{'='*70}")
        print("PHASE 1: DATA PREPARATION")
        print(f"{'='*70}")
        train_loader, val_loader, test_loader = predictor.load_and_prepare_data()
        
        print(f"\n{'='*70}")
        print("PHASE 2: MODEL TRAINING")
        print(f"{'='*70}")
        train_losses, val_losses = predictor.train(epochs=10, lr=0.0005, patience=5)
        
        print(f"\n{'='*70}")
        print("PHASE 3: SCIENTIFIC EVALUATION")
        print(f"{'='*70}")
        
        print("\nVALIDATION SET EVALUATION:")
        val_predictions, val_actuals, val_error_reduction = predictor.evaluate(
            predictor.val_loader, "Validation"
        )
        
        print("\n\nTEST SET EVALUATION (FINAL):")
        test_predictions, test_actuals, test_error_reduction = predictor.evaluate(
            predictor.test_loader, "Test"
        )
        
        print(f"\n{'='*70}")
        print("PHASE 4: BUSINESS IMPACT ANALYSIS")
        print(f"{'='*70}")
        base_improvement = 0.05
        comprehensive_report = predictor.generate_comprehensive_report(
            test_error_reduction, base_improvement
        )
        
        print(f"\n{'='*70}")
        print("PHASE 5: FORECAST GENERATION")
        print(f"{'='*70}")
        
        try:
            forecast = predictor.forecast_future_scientifically(n_days=3, use_teacher_forcing=False)
        except Exception as e:
            print(f"Forecasting failed: {e}")
            forecast = None
        
        print(f"\n{'='*70}")
        print("PHASE 6: VISUALIZATION & SAVING")
        print(f"{'='*70}")
        predictor.plot_comprehensive_results()
        predictor.save_model()
        
        print(f"\n{'='*70}")
        print("PROCESS COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        
        print(f"\nGENERATED FILES:")
        print(f"   1. comprehensive_analysis_results.png")
        print(f"   2. carbon_impact_scenarios.csv")
        print(f"   3. economic_impact_analysis.csv")
        print(f"   4. comprehensive_scientific_report.csv")
        print(f"   5. scientific_business_transformer.pth")
        
        if forecast is not None:
            print(f"   6. scientific_forecast_3days.csv")
        
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"   Validation MAE: {np.mean(np.abs(val_predictions - val_actuals)):.2f} MW")
        print(f"   Test MAE: {np.mean(np.abs(test_predictions - test_actuals)):.2f} MW")
        print(f"   Improvement vs Persistence: {test_error_reduction:.1f}%")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
