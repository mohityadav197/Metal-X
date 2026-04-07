import os, sys, torch, joblib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.ai_core.cvae_core import MetallurgicCVAE

def train_refined_model():
    df = pd.read_csv('data/augmented/augmented_data.csv')
    recipe_cols = ['time', 'temperature', 'mg', 'si', 'cu', 'fe', 'cr', 'mn', 'zn', 'ti', 'log_time', 'mg_si_ratio', 'thermal_budget']
    
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_scaled = scaler_X.fit_transform(df[recipe_cols].values)
    y_scaled = scaler_y.fit_transform(df[['yield_strength']].values)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=64, shuffle=True)

    model = MetallurgicCVAE(feature_dim=len(recipe_cols), condition_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    print("--- Training Brain on High-Quality Data ---")
    for epoch in range(150):
        model.train()
        l_total = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            rx, mu, logvar = model(bx, by)
            recon_loss = torch.nn.functional.mse_loss(rx, bx, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + (0.001 * kld_loss)
            loss.backward()
            optimizer.step()
            l_total += loss.item()
        scheduler.step()
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1} | Loss: {l_total/len(train_loader):.2f}")

    torch.save(model.state_dict(), 'models/cvae_weights.pth')
    print("--- SUCCESS: AI Model Trained ---")

if __name__ == "__main__":
    train_refined_model()