import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=30):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        return (self.X[idx:idx+self.sequence_length], 
                self.y[idx+self.sequence_length])

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        x = self.input_projection(x) * np.sqrt(self.d_model)
        
        transformer_output = self.transformer_encoder(x)
        
        last_output = transformer_output[:, -1, :]
        
        output = self.output_layer(last_output)
        return output.squeeze()

class AdvancedEnergyPredictor:
    def __init__(self, sequence_length=30, d_model=64, nhead=4, num_layers=3):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, df, target_col='energy_consumption_mwh', train_ratio=0.8):
        features = df.drop(columns=[target_col])
        target = df[target_col]
        
        X_scaled = self.scaler.fit_transform(features)
        
        split_idx = int(len(X_scaled) * train_ratio)
        
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = target[:split_idx], target[split_idx:]
        
        train_dataset = TimeSeriesDataset(X_train, y_train.values, self.sequence_length)
        test_dataset = TimeSeriesDataset(X_test, y_test.values, self.sequence_length)
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.input_dim = X_train.shape[1]
        
        return self.train_loader, self.test_loader
    
    def build_model(self):
        self.model = TransformerModel(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers
        ).to(self.device)
        
        return self.model
    
    def train(self, epochs=100, learning_rate=0.001):
        if self.model is None:
            self.build_model()
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        train_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                optimizer.zero_grad()
                
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            train_losses.append(avg_loss)
            
            scheduler.step(avg_loss)
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.6f}')
        
        return train_losses
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()

if __name__ == "__main__":
    dates = pd.date_range('2015-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'date': dates,
        'energy_consumption_mwh': 100000 + 5000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + 
                             100 * np.arange(len(dates)) / 365 + np.random.normal(0, 1000, len(dates)),
        'feature1': np.random.normal(0, 1, len(dates)),
        'feature2': np.random.normal(0, 1, len(dates))
    })
    
    predictor = AdvancedEnergyPredictor(sequence_length=30)
    train_loader, test_loader = predictor.prepare_data(sample_data)
    losses = predictor.train(epochs=50)
