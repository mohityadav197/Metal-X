import os
import logging
from dotenv import load_dotenv
# Disable ChromaDB anonymized telemetry before Chroma imports initialize.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
load_dotenv()

# Silence noisy Chroma telemetry logger errors from posthog capture incompatibilities.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from pathlib import Path
import sys
import hashlib
import math
import time
import json
from datetime import datetime, timezone
from typing import Any, List

import chromadb
import fitz
import psutil
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI, File, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from app.inference import _get_engine, run_synthesis_engine


class SynthesizeRequest(BaseModel):
    target_strength: float = Field(..., description="Desired tensile strength target")


class SynthesizeResponse(BaseModel):
    target_strength: float
    feature_order: list[str]
    features: list[float]
    feature_map: dict[str, float]
    physically_validated: bool
    physics_checks: dict[str, float | bool]
    report: str
    model_directory: str


class IntelligenceProbe(BaseModel):
    agent: str = "llama3-8b-8192"


class ResearchQueryRequest(BaseModel):
    query: str = Field(..., description="Research question or retrieval query")
    top_k: int = Field(5, ge=1, le=10)


app = FastAPI(
    title="Metallurgic-X Model Serving API",
    version="2.0.0",
    description="Production-grade backend boilerplate for CVAE and PINN inference pipelines.",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = os.path.abspath(os.environ.get("APP_ROOT", str(PROJECT_ROOT)))
CHROMA_DIR = os.path.abspath(
    os.environ.get("CHROMA_DB_PATH", os.path.join(os.getcwd(), "data", "chroma_db"))
)
CHROMA_COLLECTION_NAME = "alloy_research"
MODEL_DIR = os.path.abspath(
    os.environ.get("MODEL_DIR", os.path.join(os.getcwd(), "app", "models"))
)
INDEX_WARN_SECONDS = float(os.environ.get("RESEARCH_INDEX_WARN_SECONDS", "45"))
DEFAULT_MODEL = "llama-3.1-8b-instant"
RESEARCH_MAX_DISTANCE = float(os.environ.get("RESEARCH_MAX_DISTANCE", "1.0"))

SENTENCE_ENCODER: SentenceTransformer | None = None
SYNTHESIS_ENGINE: Any | None = None
GROQ_CLIENT: Groq | None = None
CHROMA_CLIENT: Any | None = None
GROQ_ENABLED = False
GROQ_MODEL = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
GROQ_API_KEY_MISSING_MSG = (
    "GROQ_API_KEY is not set. Expert memo generation will run in fallback mode. "
    "Set GROQ_API_KEY in your environment for Groq-powered reports."
)


def _initialize_groq_client() -> bool:
    global GROQ_CLIENT, GROQ_ENABLED

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        GROQ_ENABLED = False
        GROQ_CLIENT = None
        return False

    try:
        GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
        GROQ_ENABLED = True
        return True
    except Exception as exc:
        GROQ_ENABLED = False
        GROQ_CLIENT = None
        print(
            "GROQ client initialization failed. Running in Demo Mode. "
            f"Details: {exc}",
            file=sys.stderr,
        )
        return False


def _load_sentence_encoder() -> SentenceTransformer:
    model_name = os.environ.get("SENTENCE_ENCODER_MODEL", "all-MiniLM-L6-v2")
    memory_mode = os.environ.get("SENTENCE_ENCODER_MODE", "default").strip().lower()

    encoder = SentenceTransformer(model_name, device="cpu")

    if memory_mode in {"fp16", "half", "float16"}:
        try:
            import torch

            encoder = encoder.half()
            print("SENTENCE_ENCODER_MODE=fp16 enabled for lower memory usage.")
        except Exception as exc:
            print(
                "Sentence encoder fp16 conversion failed; continuing with fp32. "
                f"Details: {exc}",
                file=sys.stderr,
            )
    elif memory_mode in {"int8", "quantized", "quant"}:
        try:
            import torch

            first_module = encoder._first_module()
            first_module.auto_model = torch.quantization.quantize_dynamic(
                first_module.auto_model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            print("SENTENCE_ENCODER_MODE=int8 dynamic quantization enabled.")
        except Exception as exc:
            print(
                "Sentence encoder int8 quantization failed; continuing with fp32. "
                f"Details: {exc}",
                file=sys.stderr,
            )

    return encoder


def _get_chroma_collection() -> Any:
    global CHROMA_CLIENT
    os.makedirs(CHROMA_DIR, exist_ok=True)
    if CHROMA_CLIENT is None:
        CHROMA_CLIENT = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return CHROMA_CLIENT.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


def _embed_text(text: str, dim: int = 256) -> list[float]:
    if SENTENCE_ENCODER is not None:
        try:
            vector = SENTENCE_ENCODER.encode(text, normalize_embeddings=True)
            values = [float(v) for v in vector.tolist()]
            if len(values) >= dim:
                values = values[:dim]
            else:
                values.extend([0.0] * (dim - len(values)))

            norm = math.sqrt(sum(v * v for v in values))
            if norm > 0:
                values = [v / norm for v in values]
            return values
        except Exception:
            # Fall back to deterministic hashed embedding if encoder inference fails.
            pass

    vec = [0.0] * dim
    tokens = [t for t in text.lower().split() if t]
    for token in tokens[:3000]:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "big") % dim
        sign = 1.0 if (digest[2] % 2 == 0) else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return [chunk for chunk in splitter.split_text(cleaned) if chunk.strip()]


def _extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages_text: list[str] = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        full_text = "\n".join(pages_text)
        return full_text, len(doc)
    finally:
        doc.close()


def _index_pdf_document(file_name: str, pdf_bytes: bytes) -> int:
    text, page_count = _extract_pdf_text(pdf_bytes)
    _ = page_count
    chunks = _chunk_text(text)
    if not chunks:
        return 0

    collection = _get_chroma_collection()
    timestamp = int(time.time() * 1000)
    ids = [f"{file_name}-{timestamp}-{idx}" for idx in range(len(chunks))]
    embeddings = [_embed_text(chunk) for chunk in chunks]
    metadatas = [
        {
            "source": file_name,
            "chunk_index": idx,
        }
        for idx in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(chunks)


async def _index_uploaded_files(uploaded_files: List[UploadFile]) -> dict[str, Any]:
    if not uploaded_files:
        return {"status": "error", "message": "No files received for indexing."}

    total_chunks = 0
    indexed_files = 0
    warnings: list[str] = []
    errors: list[str] = []
    for up in uploaded_files:
        if not up.filename.lower().endswith(".pdf"):
            continue
        started = time.perf_counter()
        try:
            pdf_bytes = await up.read()
            chunks_added = _index_pdf_document(up.filename, pdf_bytes)
            elapsed = time.perf_counter() - started
            print(
                f"Indexed {chunks_added} chunk(s) from {up.filename} in {elapsed:.2f}s",
                file=sys.stderr,
            )

            if chunks_added > 0:
                indexed_files += 1
                total_chunks += chunks_added
            else:
                warnings.append(f"No indexable text chunks found in {up.filename}.")

            if elapsed > INDEX_WARN_SECONDS:
                warnings.append(
                    f"{up.filename} took {elapsed:.1f}s to process. "
                    "Large documents may cause slow indexing."
                )
        except MemoryError:
            errors.append(
                f"{up.filename} indexing failed: out of memory while processing PDF."
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            msg = str(exc).lower()
            if "memory" in msg or "out of memory" in msg:
                errors.append(
                    f"{up.filename} indexing failed: memory pressure detected ({exc})."
                )
            elif elapsed > INDEX_WARN_SECONDS:
                errors.append(
                    f"{up.filename} indexing failed after {elapsed:.1f}s: {exc}"
                )
            else:
                errors.append(f"{up.filename} indexing failed: {exc}")

    if errors and indexed_files == 0:
        status_text = "error"
        message_text = "Indexing failed for all uploaded documents."
    elif errors:
        status_text = "partial_success"
        message_text = "Indexing completed with some failures."
    else:
        status_text = "success"
        message_text = "Successfully indexed into the Metallurgy Knowledge Base."

    return {
        "status": status_text,
        "message": message_text,
        "files_indexed": indexed_files,
        "chunks_indexed": total_chunks,
        "collection": CHROMA_COLLECTION_NAME,
        "warnings": warnings,
        "errors": errors,
    }


def _search_research_context(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    collection = _get_chroma_collection()
    query_embedding = _embed_text(query)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(1, top_k),
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    items: list[dict[str, Any]] = []
    for idx, doc in enumerate(documents):
        distance_value = float(distances[idx]) if idx < len(distances) else None
        if distance_value is not None and distance_value > RESEARCH_MAX_DISTANCE:
            continue
        items.append(
            {
                "text": doc,
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "distance": distance_value,
            }
        )
    return items


def _unique_research_hits(hits: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    unique_items: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    for item in hits:
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        normalized = " ".join(text.split()).lower()
        if normalized in seen_texts:
            continue

        seen_texts.add(normalized)
        unique_items.append(item)
        if len(unique_items) >= limit:
            break

    return unique_items


def _deduplicate_text_blocks(blocks: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()

    for block in blocks:
        normalized = " ".join(block.split()).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(block)

    return deduped


def _build_synthesis_memo_with_context(
    feature_map: dict[str, float],
    target_strength: float,
    physically_validated: bool,
    retrieved_context: list[dict[str, Any]],
) -> str:
    sources: list[str] = []
    for item in retrieved_context:
        source = str((item.get("metadata") or {}).get("source", "")).strip()
        if source and source not in sources:
            sources.append(source)

    if sources:
        topic = ", ".join(Path(src).stem.replace("_", " ").replace("-", " ") for src in sources[:2])
    else:
        topic = "Al-Mg-Si precipitation hardening"

    context_text = "\n\n".join(item.get("text", "") for item in retrieved_context[:4]).strip()
    if not context_text:
        context_text = "No indexed research context available."

    llm_prompt = (
        "You are a Senior Metallurgical Scientist. Based on these CVAE-generated features "
        f"{json.dumps(feature_map, sort_keys=True)} and these research chunks "
        f"{context_text}, write a formal technical memo. "
        "The memo must include headers: To, From, Date, Subject, Findings, References. "
        "Findings should be 3 concise technical sentences and begin with 'Referencing documented research on'."
    )

    prompt = f"""
Role: Agentic Scientist for Al-alloy synthesis.
Write exactly 3 concise technical sentences for a formal lab memo Findings section.

Requirements:
- Start with: Referencing documented research on {topic},
- Mention whether physics checks passed.
- Mention one practical process recommendation.

Context:
{context_text}

Candidate Data:
target_strength={target_strength:.2f}
time={feature_map['time']:.4f}
temperature={feature_map['temperature']:.2f}
mg={feature_map['mg']:.4f}
si={feature_map['si']:.4f}
cu={feature_map['cu']:.4f}
fe={feature_map['fe']:.4f}
cr={feature_map['cr']:.4f}
mn={feature_map['mn']:.4f}
zn={feature_map['zn']:.4f}
ti={feature_map['ti']:.4f}
log_time={feature_map['log_time']:.4f}
mg_si_ratio={feature_map['mg_si_ratio']:.4f}
thermal_budget={feature_map['thermal_budget']:.2f}
physically_validated={physically_validated}

Answer:
""".strip()

    findings = ""
    if GROQ_CLIENT is not None:
        try:
            completion = GROQ_CLIENT.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.1,
                max_tokens=450,
                messages=[
                    {
                        "role": "system",
                        "content": "You produce concise, formal metallurgical technical memos.",
                    },
                    {
                        "role": "user",
                        "content": llm_prompt,
                    },
                ],
            )
            findings = (completion.choices[0].message.content or "").strip()
        except Exception:
            findings = ""

    fallback = "passed" if physically_validated else "requires review"
    if not findings:
        findings = (
            f"Referencing documented research on {topic}, this candidate aligns with indexed metallurgy context "
            f"and current physics status is {fallback}. Use a controlled aging window and verify Mg:Si plus "
            "Fe constraints before pilot trials."
        )

    if sources:
        references_text = "\n".join(f"- {src}" for src in sources)
    else:
        references_text = "- No indexed references available in alloy_research."

    if all(header in findings for header in ["To:", "From:", "Date:", "Subject:", "Findings:", "References:"]):
        return findings

    memo_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        "To: Alloy Development Team\n"
        "From: Metallurgic-X Agentic Scientist\n"
        f"Date: {memo_date}\n"
        f"Subject: Synthesis Findings for {target_strength:.1f} MPa Target\n\n"
        "Findings:\n"
        f"{findings}\n\n"
        "References:\n"
        f"{references_text}"
    )


def _system_intelligence_payload(agent: str = "flan-t5-small") -> dict[str, Any]:
    cpu = float(psutil.cpu_percent(interval=0.1))
    ram = float(psutil.virtual_memory().percent)
    llm_status = f"online ({agent})" if GROQ_ENABLED else "offline (missing or invalid GROQ_API_KEY)"
    return {
        "status": "online",
        "cpu_usage": cpu,
        "ram_usage": ram,
        "groq_warning": GROQ_API_KEY_MISSING_MSG if GROQ_CLIENT is None else None,
        "active_models": {
            "CVAE": "online",
            "PINN": "online",
            "GROQ-LLM": llm_status,
        },
        "physics_fidelity": "35%",
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_fail_fast_check() -> None:
    global SENTENCE_ENCODER, SYNTHESIS_ENGINE, GROQ_CLIENT, GROQ_ENABLED

    required = [
        os.path.join(MODEL_DIR, "cvae_weights.pth"),
        os.path.join(MODEL_DIR, "scaler_X.pkl"),
        os.path.join(MODEL_DIR, "scaler_y.pkl"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        print(
            "Metallurgic-X startup failed: missing required brain artifacts in ./app/models/: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure vector store exists when backend boots.
    _get_chroma_collection()

    # Warm-load core models once so request handlers avoid repeated initialization overhead.
    SYNTHESIS_ENGINE = _get_engine()
    SENTENCE_ENCODER = _load_sentence_encoder()

    if not _initialize_groq_client():
        GROQ_ENABLED = False
        GROQ_CLIENT = None
        print(GROQ_API_KEY_MISSING_MSG, file=sys.stderr)


@app.on_event("shutdown")
def shutdown_cleanup() -> None:
    global CHROMA_CLIENT, SENTENCE_ENCODER, SYNTHESIS_ENGINE, GROQ_CLIENT, GROQ_ENABLED

    if CHROMA_CLIENT is not None:
        try:
            system = getattr(CHROMA_CLIENT, "_system", None)
            stop_fn = getattr(system, "stop", None)
            if callable(stop_fn):
                stop_fn()
        except Exception as exc:
            print(f"Chroma shutdown cleanup warning: {exc}", file=sys.stderr)
        finally:
            CHROMA_CLIENT = None

    GROQ_CLIENT = None
    GROQ_ENABLED = False
    SENTENCE_ENCODER = None
    SYNTHESIS_ENGINE = None


@app.get("/")
def root() -> dict[str, str]:
    return {
        "project": "Metallurgic-X API",
        "version": "2.0.0-Stable",
        "status": "Online",
        "docs": "/docs",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/health")
def health() -> dict[str, Any]:
    cvae_loaded = SYNTHESIS_ENGINE is not None
    encoder_loaded = SENTENCE_ENCODER is not None
    overall_status = "online" if cvae_loaded and encoder_loaded else "degraded"
    return {
        "status": overall_status,
        "engine": "MetalX-v2-Production",
        "cvae_loaded": cvae_loaded,
        "sentence_encoder_loaded": encoder_loaded,
    }


@app.get("/system/intelligence")
def system_intelligence_get() -> dict[str, Any]:
    return _system_intelligence_payload(GROQ_MODEL)


@app.post("/system/intelligence")
def system_intelligence_post(payload: IntelligenceProbe) -> dict[str, Any]:
    # Backward-compatible with current UI calls while supporting new telemetry contract.
    response = _system_intelligence_payload(payload.agent)
    response["inference_ms"] = 0.0
    response["tokens_in"] = 0
    response["tokens_out"] = 0
    return response


@app.post("/research/index")
async def index_docs(files: List[UploadFile] = File(...)) -> dict[str, Any]:
    return await _index_uploaded_files(files)


@app.post("/research")
async def research(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").lower()

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_files = [item for item in form.getlist("files") if isinstance(item, UploadFile)]
        file_item = form.get("file")
        if isinstance(file_item, UploadFile):
            uploaded_files.append(file_item)
        return await _index_uploaded_files(uploaded_files)

    payload = await request.json()
    query = str(payload.get("query", "")).strip()
    _ = int(payload.get("top_k", 5))

    if not query:
        return {"status": "error", "message": "Missing query for research retrieval."}

    hits = _search_research_context(query, top_k=10)
    hits = _unique_research_hits(hits, limit=5)
    citations = []
    raw_chunks: list[str] = []
    for item in hits:
        source = (item.get("metadata") or {}).get("source")
        if source and source not in citations:
            citations.append(source)
        text = str(item.get("text", "")).strip()
        if text:
            raw_chunks.append(text)

    raw_chunks = _deduplicate_text_blocks(raw_chunks)

    joined_chunks = "\n\n".join(raw_chunks).strip()
    prompt_context = joined_chunks if joined_chunks else "No indexed context available."
    answer_prompt = (
        "You are a MetalX Metallurgy Expert. Base your answer ONLY on the provided "
        "research context. If the answer isn't there, say so. "
        f"Context: {prompt_context}"
    )

    if not os.getenv("GROQ_API_KEY"):
        return {
            "status": "error",
            "answer": "ERROR: GROQ_API_KEY is missing from environment",
            "generation_model": DEFAULT_MODEL,
            "generation_error": "ERROR: GROQ_API_KEY is missing from environment",
            "citations": citations,
            "sources": citations,
            "raw_chunks": raw_chunks,
            "retrieved_chunks": len(hits),
        }

    # Retry Groq initialization at request-time in case env was updated after boot.
    if GROQ_CLIENT is None:
        _initialize_groq_client()

    if GROQ_CLIENT is None:
        answer = "Groq initialization error: GROQ_API_KEY is missing or invalid."
    else:
        try:
            completion = GROQ_CLIENT.chat.completions.create(
                model=DEFAULT_MODEL,
                temperature=0.1,
                max_tokens=280,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful metallurgy assistant that only uses provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\n{answer_prompt}",
                    },
                ],
            )
            answer = (completion.choices[0].message.content or "").strip()
            if not answer:
                answer = "Groq generation error: empty response body from model."
        except Exception as e:
            print(f"Groq generation failed in /research: {e}", file=sys.stderr)
            answer = f"GROQ ERROR: {str(e)}"

    return {
        "status": "success",
        "answer": answer,
        "generation_model": DEFAULT_MODEL,
        "generation_error": None if not answer.startswith("GROQ ERROR:") else answer,
        "citations": citations,
        "sources": citations,
        "raw_chunks": raw_chunks,
        "retrieved_chunks": len(hits),
    }


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(payload: SynthesizeRequest) -> SynthesizeResponse:
    if SYNTHESIS_ENGINE is not None:
        synthesis_result = SYNTHESIS_ENGINE.synthesize(payload.target_strength, include_report=False)
    else:
        synthesis_result = run_synthesis_engine(payload.target_strength, include_report=False)

    if not GROQ_ENABLED:
        memo_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        synthesis_result["report"] = (
            "To: Alloy Development Team\n"
            "From: Metallurgic-X Expert Scientist (Demo Mode)\n"
            f"Date: {memo_date}\n"
            f"Subject: Demo Mode Synthesis Memo for {float(payload.target_strength):.1f} MPa Target\n\n"
            "Findings:\n"
            "Demo Mode is active because GROQ_API_KEY is missing or invalid. "
            "CVAE + PINN synthesis succeeded and produced a physically screened candidate. "
            "Set a valid GROQ_API_KEY to enable full cloud-generated expert memo enrichment.\n\n"
            "References:\n"
            "- Demo Mode: external Groq memo generation unavailable."
        )
        return SynthesizeResponse(**synthesis_result)

    # Knowledge Retrieval: fetch the two most relevant research chunks before memo generation.
    retrieval_query = (
        f"Al-Mg-Si alloy target tensile strength {payload.target_strength:.1f} MPa "
        "aging window precipitation strengthening"
    )
    retrieved = _search_research_context(retrieval_query, top_k=2)
    synthesis_result["report"] = _build_synthesis_memo_with_context(
        synthesis_result["feature_map"],
        float(payload.target_strength),
        bool(synthesis_result["physically_validated"]),
        retrieved,
    )

    return SynthesizeResponse(**synthesis_result)
