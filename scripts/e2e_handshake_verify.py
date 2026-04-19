from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


def check_backend_connectivity() -> tuple[bool, str, dict[str, Any] | None]:
    url = "http://127.0.0.1:8000/system/intelligence"
    try:
        resp = requests.get(url, timeout=2.0)
        resp.raise_for_status()
        data = resp.json()
        is_online = str(data.get("status", "")).lower() == "online"
        if not is_online:
            return False, "Backend responded but status is not online", data
        return True, "Backend connectivity OK", data
    except Exception as exc:
        return False, f"Backend connectivity failed: {exc}", None


def check_rag_model_import() -> tuple[bool, str]:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        return False, f"ModuleNotFoundError: {exc}"
    except Exception as exc:
        return False, f"Unexpected import error: {exc}"

    try:
        _ = SentenceTransformer("all-MiniLM-L6-v2")
        return True, "sentence-transformers model loaded successfully"
    except Exception as exc:
        return False, f"sentence-transformers import worked but model load failed: {exc}"


def check_frontend_timeout_contract() -> tuple[bool, str]:
    file_path = Path("Main_App.py")
    if not file_path.exists():
        return False, "Main_App.py not found"

    content = file_path.read_text(encoding="utf-8")
    expected_call = 'get_json("/system/intelligence", timeout=1.0)'
    if expected_call in content:
        return True, "Frontend timeout contract OK (1.0s in Main_App.py)"
    return False, "Frontend timeout contract missing expected 1.0s get_json call"


def main() -> int:
    results: list[dict[str, Any]] = []

    backend_ok, backend_msg, backend_data = check_backend_connectivity()
    results.append(
        {
            "check": "Backend Connectivity",
            "passed": backend_ok,
            "message": backend_msg,
            "data": backend_data,
        }
    )

    rag_ok, rag_msg = check_rag_model_import()
    results.append(
        {
            "check": "RAG Logic",
            "passed": rag_ok,
            "message": rag_msg,
        }
    )

    timeout_ok, timeout_msg = check_frontend_timeout_contract()
    results.append(
        {
            "check": "Frontend Pathing Timeout",
            "passed": timeout_ok,
            "message": timeout_msg,
        }
    )

    print(json.dumps({"results": results}, indent=2))

    all_ok = all(item["passed"] for item in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
