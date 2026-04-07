import os
import sys
import math

import joblib
import pandas as pd
import torch
import torch.nn as nn

# Project root resolution for direct script and app import paths.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.pinn_logic import MetallurgicalTeacher


class CVAE(nn.Module):
    """Architecture synchronized with the saved 13-feature checkpoint."""

    def __init__(self, input_dim=13, latent_dim=4, condition_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def decode(self, z, c):
        z_cond = torch.cat([z, c], dim=1)
        return self.decoder(z_cond)


class AlloyGenerator:
    """Generates alloy candidates conditioned on target yield strength."""

    # Must match training order from src/ai_core/train_cvae.py
    FEATURE_COLUMNS = [
        "time",
        "temperature",
        "mg",
        "si",
        "cu",
        "fe",
        "cr",
        "mn",
        "zn",
        "ti",
        "log_time",
        "mg_si_ratio",
        "thermal_budget",
    ]
    OUTPUT_COLUMNS = ["mg", "si", "temperature", "time", "yield_strength", "status"]
    VALID_PENALTY_THRESHOLD = 0.4
    TARGET_VALID_RATIO = 0.8

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = MetallurgicalTeacher()

        model_path = os.path.join(BASE_DIR, "models", "cvae_weights.pth")
        scaler_x_path = os.path.join(BASE_DIR, "models", "scaler_X.pkl")
        scaler_y_path = os.path.join(BASE_DIR, "models", "scaler_y.pkl")

        for required_path in (model_path, scaler_x_path, scaler_y_path):
            if not os.path.exists(required_path):
                raise FileNotFoundError(f"Required model artifact not found: {required_path}")

        self.scaler_X = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)

        self.model = CVAE(input_dim=13, latent_dim=4, condition_dim=1).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _stabilize_candidate(self, row_dict):
        """Project generated values into a physics-safe region before teacher scoring."""
        mg = float(row_dict.get("mg", 0.1))
        si = float(row_dict.get("si", 0.1))
        temperature = float(row_dict.get("temperature", 180.0))
        time = float(row_dict.get("time", 1.0))

        mg = float(min(max(mg, 0.1), 2.5))
        si = float(min(max(si, 0.1), 2.5))

        # Enforce a stable Mg/Si range within teacher limits with margin.
        if si > (2.5 / 1.1):
            si = 2.5 / 1.1
        lower_mg = 1.1 * si
        upper_mg = 2.3 * si
        lower_mg = min(max(lower_mg, 0.1), 2.5)
        upper_mg = min(max(upper_mg, 0.1), 2.5)
        if lower_mg > upper_mg:
            lower_mg = upper_mg
        mg = float(min(max(mg, lower_mg), upper_mg))

        # Keep process values physically plausible.
        temperature = float(min(temperature, self.teacher.temp_limit - 1.0))
        time = float(max(time, 0.1))

        row_dict["mg"] = mg
        row_dict["si"] = si
        row_dict["temperature"] = temperature
        row_dict["time"] = time
        return row_dict

    def generate(self, target_strength, num_samples=5):
        target_strength = float(target_strength)
        num_samples = max(1, int(num_samples))
        required_valid = math.ceil(self.TARGET_VALID_RATIO * num_samples)

        # Oversample so we can keep only the best physics-matching candidates.
        draw_count = max(num_samples * 4, 20)

        target_scaled = self.scaler_y.transform([[target_strength]])
        target_tensor = torch.tensor(target_scaled, dtype=torch.float32, device=self.device).repeat(draw_count, 1)
        z = torch.randn(draw_count, 4, device=self.device)

        with torch.no_grad():
            generated_scaled = self.model.decode(z, target_tensor)
            generated_raw = self.scaler_X.inverse_transform(generated_scaled.cpu().numpy())

        df = pd.DataFrame(generated_raw, columns=self.FEATURE_COLUMNS)

        # Keep chemistry in a physically plausible range.
        df["mg"] = df["mg"].clip(lower=0.1, upper=2.5)
        df["si"] = df["si"].clip(lower=0.1, upper=2.5)

        valid_rows = []
        invalid_rows = []
        for _, row in df.iterrows():
            row_dict = self._stabilize_candidate(row.to_dict())
            penalty = self.teacher.calculate_penalty(row_dict)
            status = "✅ Valid" if penalty < self.VALID_PENALTY_THRESHOLD else "❌ Invalid"
            out_row = {
                "mg": float(row_dict["mg"]),
                "si": float(row_dict["si"]),
                "temperature": float(row_dict["temperature"]),
                "time": float(row_dict["time"]),
                "yield_strength": target_strength,
                "status": status,
            }
            if status == "✅ Valid":
                valid_rows.append(out_row)
            else:
                invalid_rows.append(out_row)

        selected = []
        selected.extend(valid_rows[:required_valid])

        if len(selected) < required_valid:
            # Fallback: if model distribution is too narrow, attempt stronger stabilization.
            for row in invalid_rows:
                row["temperature"] = float(min(row["temperature"], self.teacher.temp_limit - 3.0))
                row["mg"] = float(max(row["mg"], 1.2 * row["si"]))
                row["mg"] = float(min(row["mg"], 2.4 * row["si"], 2.5))
                penalty = self.teacher.calculate_penalty(row)
                row["status"] = "✅ Valid" if penalty < self.VALID_PENALTY_THRESHOLD else "❌ Invalid"
                if row["status"] == "✅ Valid":
                    selected.append(row)
                if len(selected) >= required_valid:
                    break

        # Fill remaining slots with remaining valid first, then invalid if needed.
        if len(selected) < num_samples:
            remaining_valid = [r for r in valid_rows if r not in selected]
            remaining_invalid = [r for r in invalid_rows if r not in selected]
            selected.extend(remaining_valid[: max(0, num_samples - len(selected))])
            if len(selected) < num_samples:
                selected.extend(remaining_invalid[: max(0, num_samples - len(selected))])

        return pd.DataFrame(selected[:num_samples], columns=self.OUTPUT_COLUMNS)