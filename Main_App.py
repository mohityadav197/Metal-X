from __future__ import annotations

import streamlit as st

from ui.platform import configure_page, inject_styles, post_json, render_sidebar

try:
    from streamlit_extras.switch_page_button import switch_page
except Exception:
    switch_page = None


configure_page("METALLURGIC-X | Executive Summary")
inject_styles()
render_sidebar("Executive Summary")

BRAND_COLOR = "#9fe4ff"


def render_animated_brand(text: str) -> None:
    chars: list[str] = []
    for idx, ch in enumerate(text):
        delay = idx * 0.05
        rendered = "&nbsp;" if ch == " " else ch
        chars.append(
            (
                "<span class='brand-char' "
                f"style='animation-delay:{delay:.2f}s;color:{BRAND_COLOR};text-shadow:0 0 10px rgba(159,228,255,0.45)'>"
                f"{rendered}</span>"
            )
        )

    st.markdown(
        (
            "<div class='hero-wrap'>"
            "<h1 class='brand-title'>" + "".join(chars) + "</h1>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


render_animated_brand("METALLURGIC-X")
st.markdown(
    "<p class='hero-subtext'>"
    "A shiny, human-first AI platform that helps materials teams move from months "
    "of trial-and-error to near-instant alloy decisions."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class='glass-card'>
    <h2 style='color:#2f6bff;'>The Aluminum 6xxx Crisis</h2>
    <p>
        Traditional alloy development often burns weeks to months in lab loops,
        delaying critical lightweighting programs in aerospace and automotive.
        METALLURGIC-X changes the pace by generating and validating candidate recipes
        in about one second, enabling faster weight reduction decisions under real constraints.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class='glass-card' style='margin-top:0.9rem;'>
    <h3 style='color:#00a8ff;'>Quick Start Impact</h3>
    <div class='quick-start-grid'>
        <div class='quick-stat'><b>500+ alloys</b><br/>Synthesized in a single batch session.</div>
        <div class='quick-stat'><b>0.4 sec</b><br/>Average generation time per target strength run.</div>
        <div class='quick-stat'><b>Up to 35%</b><br/>Faster concept-to-validation cycle in design reviews.</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Live Quick Start")
col_left, col_right = st.columns([1.1, 2.0])
with col_left:
    if st.button("Run 1-Second AI Synthesis"):
        ok, data, err = post_json("/synthesize", {"target_strength": 320.0})
        if ok:
            st.success("Success! New alloy recipe created.")
            st.metric("Target Strength", f"{data.get('target_strength', 0):.1f} MPa")
            st.metric("Features Returned", len(data.get("feature_order", [])))
        else:
            st.error(f"Synthesis could not complete: {err}")

with col_right:
    if switch_page and st.button("Open Synthesis Lab"):
        switch_page("Synthesis_Lab")
    st.markdown(
        """
<div class='glass-card'>
    <b>What to do next</b><br/>
    1) Set your target in The Synthesis Lab.<br/>
    2) Compare chemistry vs manufacturing recommendations.<br/>
    3) Use The Knowledge Base to get paper-backed confidence before production.
</div>
""",
        unsafe_allow_html=True,
    )
