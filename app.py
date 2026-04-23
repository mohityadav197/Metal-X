import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.ai_core.inference import AlloyGenerator

st.set_page_config(page_title="Metallurgic-X", layout="wide", page_icon="🧬")

# Legacy UI fallback styled to match the industrial steel design language.
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #eef1f4 0%, #d9dee3 100%);
        color: #1f2933;
    }
    .industrial-title {
        color: #2d3b45;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.8);
        border-left: 4px solid #5d6d79;
        border-radius: 8px;
        padding: 16px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4b5c69, #667784);
        color: #ffffff !important;
        border-radius: 8px;
        width: 100%;
        font-weight: 600;
        height: 2.75em;
        border: 1px solid #42515c;
    }
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
    st.info("CVAE Engine: Online\n\nPhysics Teacher: Active")

st.markdown('<p class="industrial-title">METALLURGIC-X</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#4b5c69;'>Inverse Design Intelligence for Al-Mg-Si Alloys</p>", unsafe_allow_html=True)

if gen:
    if st.button("RUN SYNTHESIS"):
        results = gen.generate(target, num_samples=samples)
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        v_pct = (len(results[results['status']=='✅ Valid'])/samples)*100
        m1.metric("DESIGN GOAL", f"{target} MPa")
        m2.metric("PHYSICS MATCH", f"{v_pct:.0f}%")
        m3.metric("AVG TEMP", f"{results['temperature'].mean():.1f}°C")
        m4.metric("AVG MG %", f"{results['mg'].mean():.2f}%")

        st.subheader("Candidate Physics Report")
        st.dataframe(results[['mg', 'si', 'temperature', 'time', 'yield_strength', 'status']], width='stretch')
        
        st.subheader("Inverse Design Path Analysis")
        fig = px.parallel_coordinates(
            results,
            dimensions=['mg', 'si', 'temperature', 'time', 'yield_strength'],
            color="temperature",
            template="plotly_dark"
        )
        fig.update_traces(
            dimensions=[
                dict(label="mg", values=results["mg"], range=[0.05, 2.55]),
                dict(label="si", values=results["si"], range=[0.05, 2.55]),
                dict(label="temperature", values=results["temperature"]),
                dict(label="time", values=results["time"]),
                dict(label="yield_strength", values=results["yield_strength"]),
            ]
        )
        fig.update_layout(
            margin=dict(l=80, r=40, t=30, b=30)
        )
        st.plotly_chart(fig, width='stretch')

st.markdown("<br><hr><p style='text-align:center; color:#4b5c69;'>Mohit - Mid-Sem 2026</p>", unsafe_allow_html=True)