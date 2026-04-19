## METALLURGIC-X

AI-first platform for accelerated 6xxx aluminum alloy discovery and process recommendation.

## Project Vision

METALLURGIC-X is built to shorten the alloy development loop from months to minutes by combining generative modeling, physics-aware validation, and retrieval-augmented scientific reporting. The mission is to deliver practical, production-relevant candidate compositions and heat-treatment guidance for Al-Mg-Si (6xxx) systems.

## Tech Stack

- FastAPI: Backend orchestration and inference APIs
- Streamlit: Executive and lab-facing user interface
- ChromaDB: Persistent research vector store for RAG context
- Groq + Llama 3: Cloud-hosted expert memo generation
- Docker: Unified runtime for local validation and Railway deployment

## Architecture Overview (Hybrid Cloud)

The platform follows a hybrid inference pattern:

1. CVAE Generation (local model, in-container)
	- Generates candidate process/composition features from target strength.

2. PINN/Physics Validation (local logic, in-container)
	- Applies metallurgical constraints (for example Mg:Si window and thermal constraints) to score physical plausibility.

3. RAG Research Retrieval (local ChromaDB, persisted volume)
	- Pulls top matching research chunks from `alloy_research` in ChromaDB.
	- Persistent path is production-ready: `/app/data/chroma_db`.

4. Expert Memo (cloud inference via Groq)
	- Sends candidate features plus retrieved context to Llama 3 (`llama3-8b-8192`) for formal technical memo generation.
	- If `GROQ_API_KEY` is missing, the backend uses deterministic fallback reporting and does not crash.

## Repository Layout

- `app/main.py`: FastAPI backend, startup model warm-load, RAG pipeline, Groq memo logic
- `app/inference.py`: CVAE synthesis engine
- `ui/platform.py`: Shared frontend transport/session and backend routing
- `Main_App.py`: Main Streamlit landing and executive view
- `pages/`: Multipage Streamlit feature modules
- `data/`: Datasets and runtime data mount target
- `Dockerfile` + `start.sh`: Unified service startup for Railway

## Environment Configuration

Create a `.env` file (or configure Railway variables) with:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama3-8b-8192
BACKEND_BASE_URL=http://localhost:8000
CHROMA_DB_PATH=/app/data/chroma_db
APP_ROOT=/app
```

Notes:

- `GROQ_API_KEY` is optional for boot, but required for cloud memo generation.
- Without `GROQ_API_KEY`, the API remains online and returns fallback technical memos.

## Local Docker Run

Build image:

```bash
docker build -t metallurgic-x:latest .
```

Run container with persisted data volume:

```bash
docker run --rm -p 8501:8501 -p 8000:8000 \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  -e GROQ_MODEL=llama3-8b-8192 \
  -e CHROMA_DB_PATH=/app/data/chroma_db \
  -v metallurgicx_data:/app/data \
  metallurgic-x:latest
```

Service endpoints:

- Streamlit UI: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`

## Railway Deployment Notes

- Keep `/app/data` mounted as a persistent volume.
- Ensure `GROQ_API_KEY` is configured in Railway Variables.
- Streamlit and FastAPI run in the same container; frontend uses `http://localhost:8000` internally.

## Final Handshake Status

- Production paths are normalized and container-safe.
- ChromaDB persistence path targets `/app/data/chroma_db`.
- Groq integration is environment-hardened with non-crashing fallback behavior.
