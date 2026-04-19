from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import requests
import streamlit as st

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
BACKEND_OFFLINE_MESSAGE = "Backend Engine Offline - Please start the FastAPI server."
REQUEST_TIMEOUT_MESSAGE = "Request timed out."
STYLE_PATH = Path(__file__).resolve().parent.parent / "style.css"
BACKEND_SESSION = requests.Session()


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, page_icon="MX", layout="wide")


def inject_styles() -> None:
    if STYLE_PATH.exists():
        css = STYLE_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_sidebar(active_page: str) -> None:
    # Keep native page navigation and only append engine status context.
    _ = active_page
    ok, _, _ = get_json("/system/intelligence", timeout=2.0)
    if ok:
        status_dot = "online"
        status_text = "Engine Online"
    else:
        status_dot = "warning"
        status_text = "Engine Standby"

    st.sidebar.markdown(
        (
            "<div class='sidebar-engine'>"
            f"<span class='status-dot {status_dot}'></span>Backend: {status_text}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _normalize_backend_error(exc: requests.RequestException) -> str:
    if isinstance(exc, requests.ConnectionError):
        return BACKEND_OFFLINE_MESSAGE
    if isinstance(exc, requests.Timeout):
        return REQUEST_TIMEOUT_MESSAGE
    return str(exc)


def get_json(endpoint: str, timeout: float = 2.0) -> tuple[bool, Any, str | None]:
    url = f"{BACKEND_BASE_URL}{endpoint}"
    try:
        response = BACKEND_SESSION.get(url, timeout=timeout)
        response.raise_for_status()
        return True, response.json(), None
    except requests.RequestException as exc:
        return False, None, _normalize_backend_error(exc)


def post_json(endpoint: str, payload: dict[str, Any], timeout: float = 2.0) -> tuple[bool, Any, str | None]:
    url = f"{BACKEND_BASE_URL}{endpoint}"
    try:
        response = BACKEND_SESSION.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return True, response.json(), None
    except requests.RequestException as exc:
        return False, None, _normalize_backend_error(exc)


def post_form(
    endpoint: str,
    data: dict[str, Any],
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    timeout: float = 2.0,
) -> tuple[bool, Any, str | None]:
    url = f"{BACKEND_BASE_URL}{endpoint}"
    try:
        response = BACKEND_SESSION.post(url, data=data, files=files, timeout=timeout)
        response.raise_for_status()
        if response.headers.get("content-type", "").startswith("application/json"):
            return True, response.json(), None
        return True, {"message": response.text}, None
    except requests.RequestException as exc:
        return False, None, _normalize_backend_error(exc)
