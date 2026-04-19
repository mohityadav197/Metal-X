from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.platform import BACKEND_OFFLINE_MESSAGE, configure_page, inject_styles, post_json, render_sidebar


configure_page("METALLURGIC-X | Synthesis Lab")
inject_styles()
render_sidebar("The Synthesis Lab")

st.markdown("## The Synthesis Lab")
st.caption("Generate and review alloy features with clear integrity checks.")

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("Target Control")
target_strength = st.slider("Target Strength (MPa)", min_value=150, max_value=500, value=320)
execute = st.button("Execute Synthesis")
st.markdown("</div>", unsafe_allow_html=True)

if execute:
    ok, data, err = post_json("/synthesize", {"target_strength": float(target_strength)})
    if not ok:
        st.error(BACKEND_OFFLINE_MESSAGE if err == BACKEND_OFFLINE_MESSAGE else f"Request failed: {err}")
    else:
        feature_order = data.get("feature_order", [])
        features = data.get("features", [])
        feature_map = data.get("feature_map", {})
        physically_validated = bool(data.get("physically_validated", False))

        if not feature_order and feature_map:
            feature_order = list(feature_map.keys())
        if not features and feature_order:
            features = [feature_map.get(k, 0.0) for k in feature_order]

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Generated 13-Feature Recipe")
        if feature_order and features:
            table_df = pd.DataFrame(
                {
                    "Feature": feature_order,
                    "Value": [float(v) for v in features],
                }
            )
            st.dataframe(table_df, use_container_width=True, hide_index=True)
        else:
            st.info("No feature vector returned by backend.")

        status_text = "PASSED" if physically_validated else "REVIEW"
        st.write(f"Physics Validation: {status_text}")

        report = data.get("report", "No scientist memo was returned.")
        st.markdown("<h4 style='margin-top:1rem;'>Agentic Scientist Memo</h4>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='glass-card memo-box'>{report}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        """
<div class='glass-card'>
    <h4>Ready for Execution</h4>
    Set the target strength and click <b>Execute Synthesis</b> to generate a standard 13-feature table and report.
</div>
""",
        unsafe_allow_html=True,
    )

with st.expander("View Physics Integrity Rules"):
    st.write("The system balances Magnesium and Silicon content to ensure optimal structural strengthening.")
    st.write("Iron levels are strictly capped at 0.5% to prevent material brittleness.")
    st.write("Process temperatures are cross-checked against standard 6xxx-series solubility limits.")
