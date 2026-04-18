from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.inference import run_synthesis_engine


class SynthesizeRequest(BaseModel):
    target_strength: float = Field(..., description="Desired tensile strength target")


class SynthesizeResponse(BaseModel):
    target_strength: float
    feature_order: list[str]
    features: list[float]
    feature_map: dict[str, float]
    physically_validated: bool
    physics_checks: dict[str, float | bool]
    report: str
    model_directory: str


app = FastAPI(
    title="Metallurgic-X Model Serving API",
    version="2.0.0",
    description="Production-grade backend boilerplate for CVAE and PINN inference pipelines.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_fail_fast_check() -> None:
    model_dir = Path(__file__).resolve().parent / "models"
    required = [
        model_dir / "cvae_weights.pth",
        model_dir / "scaler_X.pkl",
        model_dir / "scaler_y.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(
            "Metallurgic-X startup failed: missing required brain artifacts in ./app/models/: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "online", "engine": "MetalX-v2-Production"}


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(payload: SynthesizeRequest) -> SynthesizeResponse:
    synthesis_result = run_synthesis_engine(payload.target_strength)
    return SynthesizeResponse(**synthesis_result)
