from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
import streamlit as st

BACKEND_BASE_URL = "http://127.0.0.1:8000"
STYLE_PATH = Path(__file__).resolve().parent.parent / "style.css"


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, page_icon="MX", layout="wide")


def inject_styles() -> None:
    if STYLE_PATH.exists():
        css = STYLE_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_sidebar(active_page: str) -> None:
    st.sidebar.markdown("<div class='side-title'>METALLURGIC-X</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='side-subtitle'>Shiny Alloy Intelligence Platform</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        (
            "<div class='glass-card side-note'>"
            f"<b>Current View:</b> {active_page}<br/>"
            "Backend: http://127.0.0.1:8000"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def post_json(endpoint: str, payload: dict[str, Any], timeout: float = 45.0) -> tuple[bool, Any, str | None]:
    url = f"{BACKEND_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return True, response.json(), None
    except requests.RequestException as exc:
        return False, None, str(exc)


def post_form(
    endpoint: str,
    data: dict[str, Any],
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    timeout: float = 90.0,
) -> tuple[bool, Any, str | None]:
    url = f"{BACKEND_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, data=data, files=files, timeout=timeout)
        response.raise_for_status()
        if response.headers.get("content-type", "").startswith("application/json"):
            return True, response.json(), None
        return True, {"message": response.text}, None
    except requests.RequestException as exc:
        return False, None, str(exc)
