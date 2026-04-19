from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.platform import BACKEND_OFFLINE_MESSAGE, configure_page, get_json, inject_styles, render_sidebar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SEARCH_DIRS = [PROJECT_ROOT / "models", PROJECT_ROOT / "app" / "models"]
REQUIRED_MODEL_ARTIFACTS = ["cvae_weights.pth", "scaler_X.pkl", "scaler_y.pkl"]


def _missing_model_artifacts() -> list[str]:
    missing: list[str] = []
    for artifact in REQUIRED_MODEL_ARTIFACTS:
        if not any((base / artifact).exists() for base in MODEL_SEARCH_DIRS):
            missing.append(artifact)
    return missing


configure_page("METALLURGIC-X | Architecture Blueprint")
inject_styles()
render_sidebar("Architecture Blueprint")

st.markdown("## System Intelligence")
st.caption("Simple visual explanation of how METALLURGIC-X turns goals into decisions")

st.markdown(
    """
<div class='glass-card'>
    <h3>How It Works, Step by Step</h3>
    <div class='blueprint-grid'>
        <div class='blueprint-block'>[User Target]<div class='blueprint-caption'>You set the target.</div></div>
        <div class='blueprint-arrow'>➔</div>
        <div class='blueprint-block'>[CVAE Brain]<div class='blueprint-caption'>The AI imagines.</div></div>
        <div class='blueprint-arrow'>➔</div>
        <div class='blueprint-block'>[Physics Guard]<div class='blueprint-caption'>The Laws of Physics filter.</div></div>
        <div class='blueprint-arrow'>➔</div>
        <div class='blueprint-block'>[Expert Report]<div class='blueprint-caption'>The Scientist explains.</div></div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Live Pulse")
col_a, col_b = st.columns([1, 1.5], gap="large")

with col_a:
    if st.button("Check System Pulse"):
        missing_artifacts = _missing_model_artifacts()
        if missing_artifacts:
            st.error("Model Weights: Missing")
            st.caption("Missing artifacts: " + ", ".join(missing_artifacts))
            st.caption("Searched in: " + ", ".join(str(path) for path in MODEL_SEARCH_DIRS))
        else:
            st.success("Model Weights: Present")

        ok, data, err = get_json("/system/intelligence", timeout=2.0)
        if ok:
            st.success("Architecture services are active.")
            st.metric("CPU Usage", f"{float(data.get('cpu_usage', 0.0)):.1f}%")
            st.metric("RAM Usage", f"{float(data.get('ram_usage', 0.0)):.1f}%")
        else:
            st.error(BACKEND_OFFLINE_MESSAGE if err == BACKEND_OFFLINE_MESSAGE else f"Request failed: {err}")

with col_b:
    st.info(
        "Why this matters: Instead of guessing in the lab for weeks, you can use one clear digital flow "
        "with target input, AI generation, physics filtering, and a readable expert summary."
    )
