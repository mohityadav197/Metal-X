from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import torch
from transformers import pipeline

from src.ai_core.cvae_core import MetallurgicCVAE
from src.pinn_logic import MetallurgicalTeacher


APP_DIR = Path(__file__).parent
APP_MODEL_DIR = Path(__file__).parent / "models"

# Must stay synchronized with training schema order.
FEATURE_ORDER = [
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


@lru_cache(maxsize=1)
def _get_explainer() -> Any:
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
    )


def generate_report(feature_map: dict[str, float], target_strength: float, physically_validated: bool) -> str:
    """Generate a concise metallurgical explanation with atomic few-shot ICL."""
    few_shot_prompt = f"""
Role: Agentic Scientist for Al-alloy synthesis. Output 2-3 technical sentences.
Rules: Reference Mg:Si window (1.0-2.5), thermal boundary (<=195C), and one process action.

Ex1 | data:t=6.8,T=182,mg=1.42,si=0.86,cu=0.21,fe=0.17,cr=0.09,mn=0.33,zn=0.14,ti=0.04,lt=1.92,r=1.65,b=1237.6 | y=312 | ok=True
Ex1 -> Mg:Si is centered in the stable region and temperature is below limit, so precipitation kinetics are credible. Maintain dwell near 6-7h for repeatability.

Ex2 | data:t=7.5,T=196.5,mg=1.60,si=0.72,cu=0.25,fe=0.15,cr=0.08,mn=0.36,zn=0.12,ti=0.05,lt=2.01,r=2.22,b=1473.8 | y=328 | ok=False
Ex2 -> Ratio is acceptable but thermal boundary is violated, increasing over-aging risk. Lower temperature or reduce soak time to recover physical validity.

Ex3 | data:t=5.4,T=176,mg=0.94,si=0.95,cu=0.19,fe=0.16,cr=0.07,mn=0.28,zn=0.10,ti=0.03,lt=1.69,r=0.99,b=950.4 | y=295 | ok=False
Ex3 -> Thermal condition is safe, but Mg:Si is below stoichiometric window and likely under-supplies strengthening phases. Increase Mg slightly or decrease Si.

Now analyze:
data:t={feature_map['time']:.4f},T={feature_map['temperature']:.2f},mg={feature_map['mg']:.4f},si={feature_map['si']:.4f},cu={feature_map['cu']:.4f},fe={feature_map['fe']:.4f},cr={feature_map['cr']:.4f},mn={feature_map['mn']:.4f},zn={feature_map['zn']:.4f},ti={feature_map['ti']:.4f},lt={feature_map['log_time']:.4f},r={feature_map['mg_si_ratio']:.4f},b={feature_map['thermal_budget']:.2f} | y={target_strength:.2f} | ok={physically_validated}
Answer:
""".strip()

    explainer = _get_explainer()
    output = explainer(
        few_shot_prompt,
        temperature=0.1,
        top_p=0.9,
        max_new_tokens=80,
        do_sample=True,
    )
    return str(output[0]["generated_text"]).strip()


class _SynthesisEngine:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.teacher = MetallurgicalTeacher()
        self.model_dir = APP_MODEL_DIR

        self.model_path = Path(__file__).parent / "models" / "cvae_weights.pth"
        self.scaler_x_path = Path(__file__).parent / "models" / "scaler_X.pkl"
        self.scaler_y_path = Path(__file__).parent / "models" / "scaler_y.pkl"

        self._validate_model_artifacts()

        self.scaler_X = joblib.load(self.scaler_x_path)
        self.scaler_y = joblib.load(self.scaler_y_path)

        self.model = MetallurgicCVAE(feature_dim=len(FEATURE_ORDER), condition_dim=1, latent_dim=4).to("cpu")
        state_dict = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _validate_model_artifacts(self) -> None:
        required = [self.model_path, self.scaler_x_path, self.scaler_y_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Model artifacts not found in ./app/models/: " + ", ".join(missing)
            )

    def _apply_postprocessing(self, feature_map: dict[str, float]) -> dict[str, float]:
        # Keep minimal physical clipping before PINN scoring.
        feature_map["si"] = float(max(feature_map["si"], 1e-4))
        feature_map["mg"] = float(max(feature_map["mg"], 1e-4))
        feature_map["time"] = float(max(feature_map["time"], 1e-4))
        feature_map["temperature"] = float(max(feature_map["temperature"], 1e-4))

        # Derived process descriptors should remain synchronized with chemistry/process values.
        feature_map["log_time"] = float(torch.log(torch.tensor(feature_map["time"] + 1e-6)).item())
        feature_map["mg_si_ratio"] = float(feature_map["mg"] / (feature_map["si"] + 1e-6))
        feature_map["thermal_budget"] = float(feature_map["temperature"] * feature_map["time"])
        return feature_map

    def synthesize(self, target_strength: float) -> dict[str, Any]:
        target_strength = float(target_strength)

        target_scaled = self.scaler_y.transform([[target_strength]])
        target_tensor = torch.tensor(target_scaled, dtype=torch.float32).to("cpu")
        z = torch.randn(1, 4, device="cpu")

        with torch.no_grad():
            decoded_scaled = self.model.decoder(torch.cat([z, target_tensor], dim=1))
            decoded_raw = self.scaler_X.inverse_transform(decoded_scaled.cpu().numpy())[0]

        feature_map = {
            name: float(decoded_raw[idx]) for idx, name in enumerate(FEATURE_ORDER)
        }
        feature_map = self._apply_postprocessing(feature_map)

        mg_si_ratio = float(feature_map["mg"] / (feature_map["si"] + 1e-6))
        thermal_ok = bool(feature_map["temperature"] <= self.teacher.temp_limit)
        ratio_ok = bool(1.0 <= mg_si_ratio <= 2.5)
        penalty = float(self.teacher.calculate_penalty(feature_map))
        physically_validated = bool(ratio_ok and thermal_ok and penalty < 0.4)

        return {
            "target_strength": target_strength,
            "feature_order": FEATURE_ORDER,
            "features": [feature_map[name] for name in FEATURE_ORDER],
            "feature_map": feature_map,
            "physically_validated": physically_validated,
            "physics_checks": {
                "mg_si_ratio": mg_si_ratio,
                "ratio_ok": ratio_ok,
                "thermal_boundary_ok": thermal_ok,
                "pinn_penalty": penalty,
                "temperature_limit_c": float(self.teacher.temp_limit),
            },
            "report": generate_report(feature_map, target_strength, physically_validated),
            "model_directory": str(self.model_dir),
        }


@lru_cache(maxsize=1)
def _get_engine() -> _SynthesisEngine:
    return _SynthesisEngine()


def run_synthesis_engine(target_strength: float) -> dict[str, Any]:
    return _get_engine().synthesize(target_strength)
