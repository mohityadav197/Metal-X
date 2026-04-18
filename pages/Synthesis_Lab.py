from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.platform import configure_page, inject_styles, post_json, render_sidebar


configure_page("METALLURGIC-X | Synthesis Lab")
inject_styles()
render_sidebar("The Synthesis Lab")


CHEMISTRY_KEYS = {
    "mg",
    "si",
    "fe",
    "cu",
    "mn",
    "cr",
    "zn",
    "ti",
    "al",
}


def classify_feature(name: str) -> tuple[str, str]:
    normalized = name.lower().strip()
    if normalized in CHEMISTRY_KEYS or normalized.endswith("%"):
        return "Chemistry", "Chemistry"
    return "Manufacturing", "Manufacturing"


def style_feature_table(df: pd.DataFrame):
    max_value = max(float(df["Value"].max()), 1e-8)

    def row_style(row: pd.Series) -> list[str]:
        intensity = float(row["Value"]) / max_value
        blue_alpha = 0.22 + (0.66 * intensity)
        glow_alpha = 0.16 + (0.28 * intensity)
        if row["Category"] == "Chemistry":
            bg = f"background: linear-gradient(90deg, rgba(0, 112, 182, 0.45), rgba(0, 168, 255, {blue_alpha:.3f}))"
            border = "border: 1px solid rgba(182, 255, 46, 0.45)"
        else:
            bg = f"background: linear-gradient(90deg, rgba(0, 86, 146, 0.45), rgba(0, 148, 232, {blue_alpha:.3f}))"
            border = "border: 1px solid rgba(0, 168, 255, 0.5)"
        shadow = f"box-shadow: inset 0 0 0 9999px rgba(0, 168, 255, {glow_alpha:.3f})"
        return [f"{bg};{border};{shadow};color:#e8f7ff"] * len(row)

    return (
        df.style.apply(row_style, axis=1)
        .format({"Value": "{:.4f}"})
        .set_properties(**{"font-size": "16px", "font-weight": "700"})
    )

st.markdown("## The Synthesis Lab")
st.caption("Friendly inverse design experience with clear category-based results")

st.markdown(
    """
<div class='crumb-wrap'>
    <span class='crumb-step'>Target Input</span>
    <span class='crumb-arrow'>⮕</span>
    <span class='crumb-step'>AI Generation</span>
    <span class='crumb-arrow'>⮕</span>
    <span class='crumb-step'>Physics Check</span>
    <span class='crumb-arrow'>⮕</span>
    <span class='crumb-step'>Result</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class='glass-card'>
    <h3 style='color:#2f6bff;'>Inverse Design Path</h3>
    <div class='path-wrap'>
        <div class='path-step'>Target Strength</div>
        <div class='path-step'>CVAE Synthesis</div>
        <div class='path-step'>PINN Physics Guard</div>
        <div class='path-step'>Final Recipe</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Target Control")
    target_strength = st.slider("Target Strength (MPa)", min_value=150, max_value=500, value=320)
    execute = st.button("Execute Synthesis")
    st.markdown("</div>", unsafe_allow_html=True)

if execute:
    ok, data, err = post_json("/synthesize", {"target_strength": float(target_strength)})
    if not ok:
        with right_col:
            st.error(f"Synthesis request failed: {err}")
    else:
        feature_order = data.get("feature_order", [])
        features = data.get("features", [])
        feature_map = data.get("feature_map", {})
        physically_validated = bool(data.get("physically_validated", False))

        if not feature_order and feature_map:
            feature_order = list(feature_map.keys())
        if not features and feature_order:
            features = [feature_map.get(k, 0.0) for k in feature_order]

        with right_col:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Colorful 13-Feature Recipe Table")
            if feature_order and features:
                table_rows = []
                for name, value in zip(feature_order, features):
                    category, label = classify_feature(name)
                    icon = "🧪" if category == "Chemistry" else "⚙️"
                    table_rows.append(
                        {
                            "Feature": name,
                            "Value": float(value),
                            "Category": label,
                            "Icon": icon,
                        }
                    )

                table_df = pd.DataFrame(table_rows)
                st.dataframe(
                    style_feature_table(table_df[["Icon", "Feature", "Category", "Value"]]),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown(
                    "<span class='chem-chip'>🧪 Chemistry</span> "
                    "<span class='manu-chip'>⚙️ Manufacturing</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No feature vector returned by backend.")

            pill_class = "pill-ok" if physically_validated else "pill-warn"
            pill_text = "Physics Validation: PASSED" if physically_validated else "Physics Validation: REVIEW"
            st.markdown(
                f"<span class='pill {pill_class}'>{pill_text}</span>",
                unsafe_allow_html=True,
            )

            report = data.get("report", "No scientist memo was returned.")
            st.markdown("<h4 style='margin-top:1rem;'>Agentic Scientist Memo</h4>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='glass-card memo-box'>{report}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
else:
    with right_col:
        st.markdown(
            """
<div class='glass-card'>
    <h4>Ready for Execution</h4>
    Set the target strength and click <b>Execute Synthesis</b> to generate
    a colorful 13-feature recipe table plus a validation badge and memo.
</div>
""",
            unsafe_allow_html=True,
        )

with st.expander("The Physics Guardrails", expanded=False):
    st.markdown(
        """
<div class='glass-card'>
    <h4 style='margin-top:0;color:#00a8ff;'>PINN Constraint Rules</h4>
    <p><b>Mg:Si Stoichiometry</b>: $1.73$ ratio enforcement for balanced precipitation pathways.</p>
    <p><b>Solubility Limits</b>: Maximum wt% constraints at $580$°C to avoid unstable phase behavior.</p>
    <p><b>Thermal Budget</b>: $T \times \log(t)$ aging laws constrain feasible heat-treatment schedules.</p>
</div>
""",
        unsafe_allow_html=True,
    )
