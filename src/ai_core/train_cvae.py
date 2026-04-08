import os
import sys
import shutil
from datetime import datetime

import torch
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.ai_core.cvae_core import MetallurgicCVAE


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _backup_if_exists(file_paths, backup_root):
    _ensure_dir(backup_root)
    copied = []
    for src in file_paths:
        if os.path.exists(src):
            dst = os.path.join(backup_root, os.path.basename(src))
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def _compute_loss_terms(model, xb, yb):
    rx, mu, logvar = model(xb, yb)
    recon_loss = torch.nn.functional.mse_loss(rx, xb, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + (0.001 * kld_loss)
    return total_loss, recon_loss, kld_loss


def train_refined_model(promote_to_production=False):
    df = pd.read_csv('data/augmented/augmented_data.csv')
    recipe_cols = ['time', 'temperature', 'mg', 'si', 'cu', 'fe', 'cr', 'mn', 'zn', 'ti', 'log_time', 'mg_si_ratio', 'thermal_budget']

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_scaled = scaler_X.fit_transform(df[recipe_cols].values)
    y_scaled = scaler_y.fit_transform(df[['yield_strength']].values)

    models_dir = 'models'
    _ensure_dir(models_dir)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    candidate_model_path = os.path.join(models_dir, f'cvae_weights_candidate_{timestamp}.pth')
    candidate_scaler_x_path = os.path.join(models_dir, f'scaler_X_candidate_{timestamp}.pkl')
    candidate_scaler_y_path = os.path.join(models_dir, f'scaler_y_candidate_{timestamp}.pkl')
    metrics_path = os.path.join(models_dir, f'training_metrics_{timestamp}.csv')

    production_model_path = os.path.join(models_dir, 'cvae_weights.pth')
    production_scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
    production_scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')

    backup_root = os.path.join(models_dir, 'backups', timestamp)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=256, shuffle=False)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=64, shuffle=True)

    model = MetallurgicCVAE(feature_dim=len(recipe_cols), condition_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    history = []
    best_val_total = float('inf')
    best_state_dict = None

    print("--- Training Brain on High-Quality Data (with validation tracking) ---")
    for epoch in range(150):
        model.train()
        train_total_sum = 0.0
        train_recon_sum = 0.0
        train_kld_sum = 0.0
        train_samples = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            total_loss, recon_loss, kld_loss = _compute_loss_terms(model, bx, by)
            total_loss.backward()
            optimizer.step()

            batch_size = bx.size(0)
            train_total_sum += total_loss.item()
            train_recon_sum += recon_loss.item()
            train_kld_sum += kld_loss.item()
            train_samples += batch_size

        model.eval()
        val_total_sum = 0.0
        val_recon_sum = 0.0
        val_kld_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for bx, by in val_loader:
                total_loss, recon_loss, kld_loss = _compute_loss_terms(model, bx, by)
                batch_size = bx.size(0)
                val_total_sum += total_loss.item()
                val_recon_sum += recon_loss.item()
                val_kld_sum += kld_loss.item()
                val_samples += batch_size

        train_total = train_total_sum / max(train_samples, 1)
        train_recon = train_recon_sum / max(train_samples, 1)
        train_kld = train_kld_sum / max(train_samples, 1)
        val_total = val_total_sum / max(val_samples, 1)
        val_recon = val_recon_sum / max(val_samples, 1)
        val_kld = val_kld_sum / max(val_samples, 1)
        val_gap = val_total - train_total

        history.append(
            {
                'epoch': epoch + 1,
                'train_total': train_total,
                'train_recon': train_recon,
                'train_kld': train_kld,
                'val_total': val_total,
                'val_recon': val_recon,
                'val_kld': val_kld,
                'val_gap': val_gap,
                'lr': optimizer.param_groups[0]['lr'],
            }
        )

        if val_total < best_val_total:
            best_val_total = val_total
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Total: {train_total:.4f} | "
                f"Val Total: {val_total:.4f} | "
                f"Gap: {val_gap:.4f}"
            )

    # Save overfitting/fit-quality matrix for reporting.
    pd.DataFrame(history).to_csv(metrics_path, index=False)

    # Always save candidate artifacts first; this preserves the current happy state.
    if best_state_dict is not None:
        torch.save(best_state_dict, candidate_model_path)
    else:
        torch.save(model.state_dict(), candidate_model_path)
    joblib.dump(scaler_X, candidate_scaler_x_path)
    joblib.dump(scaler_y, candidate_scaler_y_path)

    print(f"--- Candidate model saved: {candidate_model_path}")
    print(f"--- Candidate scalers saved: {candidate_scaler_x_path}, {candidate_scaler_y_path}")
    print(f"--- Training metrics saved: {metrics_path}")

    if promote_to_production:
        copied = _backup_if_exists(
            [production_model_path, production_scaler_x_path, production_scaler_y_path],
            backup_root,
        )
        shutil.copy2(candidate_model_path, production_model_path)
        shutil.copy2(candidate_scaler_x_path, production_scaler_x_path)
        shutil.copy2(candidate_scaler_y_path, production_scaler_y_path)

        print("--- Production artifacts updated from candidate.")
        if copied:
            print(f"--- Backup created: {backup_root}")

    print("--- SUCCESS: AI Model Trained ---")

if __name__ == "__main__":
    train_refined_model()