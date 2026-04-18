from __future__ import annotations

import streamlit as st

from ui.platform import configure_page, inject_styles, post_json, render_sidebar


configure_page("METALLURGIC-X | Architecture Blueprint")
inject_styles()
render_sidebar("Architecture Blueprint")

st.markdown("## Architecture Blueprint")
st.caption("Simple visual explanation of how METALLURGIC-X turns goals into decisions")

st.markdown(
    """
<div class='glass-card'>
    <h3 style='color:#2f6bff;'>How It Works, Step by Step</h3>
    <div class='blueprint-grid'>
        <div class='blueprint-block'>[Target]<div class='blueprint-caption'>You set the target.</div></div>
        <div class='blueprint-arrow'>⟶</div>
        <div class='blueprint-block'>[CVAE Brain]<div class='blueprint-caption'>The Brain dreams it.</div></div>
        <div class='blueprint-arrow'>⟶</div>
        <div class='blueprint-block'>[Physics Guard]<div class='blueprint-caption'>The Laws of Physics check it.</div></div>
        <div class='blueprint-arrow'>⟶</div>
        <div class='blueprint-block'>[Expert Report]<div class='blueprint-caption'>The Scientist explains it.</div></div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Live Pulse")
col_a, col_b = st.columns([1, 1.5], gap="large")

with col_a:
    if st.button("Check System Pulse"):
        ok, data, err = post_json("/system/intelligence", {"agent": "flan-t5-small"}, timeout=30.0)
        if ok:
            st.success("Architecture services are active.")
            st.metric("Inference Time", f"{float(data.get('inference_ms', 0.0)):.1f} ms")
            st.metric("Tokens Used", int(data.get("tokens_in", 0)) + int(data.get("tokens_out", 0)))
        else:
            if err and "404" in err:
                st.info("System Initializing")
            else:
                st.info("System Initializing")

with col_b:
    st.markdown(
        """
<div class='glass-card'>
    <h4 style='color:#00a8ff;'>Why this matters</h4>
    <p>
        Instead of guessing in the lab for weeks, you can see one clear digital flow:
        target goal in, intelligent recipe out, with physics safety checks and an easy report.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )
