#!/bin/sh
set -eu

# Start FastAPI internally for Streamlit-to-backend calls.
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Give backend time to load models before UI traffic starts.
sleep 5

# Keep Streamlit as the foreground process (Railway entrypoint).
streamlit run Main_App.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true

# If Streamlit exits, stop FastAPI as well.
kill "$FASTAPI_PID" >/dev/null 2>&1 || true
