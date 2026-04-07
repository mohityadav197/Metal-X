import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.ai_core.inference import AlloyGenerator

st.set_page_config(page_title="Metallurgic-X", layout="wide", page_icon="🧬")

# --- PREMIUM CSS ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #0d1117, #010409); color: #e6edf3; }
    .neon-title { color: #00d4ff; text-shadow: 0 0 15px #00d4ff; font-size: 3.5rem; font-weight: 800; text-align: center; }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.03); border-left: 5px solid #00d4ff; border-radius: 10px; padding: 20px; }
    .stButton>button { background: linear-gradient(90deg, #00d4ff, #005f73); color: white !important; border-radius: 50px; width: 100%; font-weight: bold; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return AlloyGenerator()

gen = load_engine()

# --- SIDEBAR ---
with st.sidebar:
    st.header("COMMAND CENTER")
    target = st.slider("TARGET STRENGTH (MPa)", 150, 450, 320)
    samples = st.slider("VARIANTS", 1, 10, 5)
    st.markdown("---")
    st.info("🟢 CVAE Engine: Online\n\n🟢 Physics Teacher: Active")

st.markdown('<p class="neon-title">METALLURGIC-X</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#8b949e;'>Inverse Design Intelligence for Al-Mg-Si Alloys</p>", unsafe_allow_html=True)

if gen:
    if st.button("INITIATE NEURAL SYNTHESIS"):
        results = gen.generate(target, num_samples=samples)
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        v_pct = (len(results[results['status']=='✅ Valid'])/samples)*100
        m1.metric("DESIGN GOAL", f"{target} MPa")
        m2.metric("PHYSICS MATCH", f"{v_pct:.0f}%")
        m3.metric("AVG TEMP", f"{results['temperature'].mean():.1f}°C")
        m4.metric("AVG MG %", f"{results['mg'].mean():.2f}%")

        st.subheader("🧬 Candidate Physics Report")
        st.dataframe(results[['mg', 'si', 'temperature', 'time', 'yield_strength', 'status']], use_container_width=True)
        
        st.subheader("🕸️ Inverse Design Path Analysis")
        fig = px.parallel_coordinates(
            results,
            dimensions=['mg', 'si', 'temperature', 'time', 'yield_strength'],
            color="temperature",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br><hr><p style='text-align:center; color:#30363d;'>Mohit - Mid-Sem 2026</p>", unsafe_allow_html=True)