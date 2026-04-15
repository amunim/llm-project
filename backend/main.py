"""
FastAPI backend for the NUST Bank RAG system.

Environment variables:
  MODEL_PATH       - local path to the GGUF file (highest priority)
  HF_HUB_REPO      - HuggingFace Hub repo id  (e.g. zain-0/nust-bank-qwen2.5-3b-gguf)
  HF_HUB_FILENAME  - filename inside the repo  (e.g. nust_bank_qwen2.5_3b_q4km.gguf)
  HF_TOKEN         - HuggingFace read token (optional for public repos)
  DB_PATH          - override ChromaDB location (default: data/chroma_db)

Vector store:
  POST /vectorstore/reset   — clear the Chroma collection (empty index)
  POST /vectorstore/ingest  — multipart files: .pdf, .docx, .json, .txt (or matching Content-Type)

Local test:
  $env:MODEL_PATH = "models/nust_bank_qwen2.5_3b_q4km.gguf"
  uvicorn backend.main:app --reload --port 8000
  # POST http://localhost:8000/query  {"question": "What is the Little Champs Account?"}
"""

import io
import json
import os
import zipfile

# Before any stack that loads OpenMP (sentence-transformers, llama.cpp, etc.)
from src.env_bootstrap import ensure_valid_thread_env

ensure_valid_thread_env()

import re
import logging
from contextlib import asynccontextmanager

from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nust_bank_api")

# ---------------------------------------------------------------------------
# Global pipeline holder
# ---------------------------------------------------------------------------
_pipeline = None


def _resolve_model_path() -> str:
    """Return local GGUF path, downloading from HF Hub if needed."""
    # 1. Explicit local override
    model_path = os.getenv("MODEL_PATH", "").strip()
    if model_path and os.path.isfile(model_path):
        logger.info("Using local model: %s", model_path)
        return model_path

    # 2. Download from HF Hub
    repo_id = os.getenv("HF_HUB_REPO", "").strip()
    filename = os.getenv("HF_HUB_FILENAME", "").strip()
    if not repo_id or not filename:
        raise RuntimeError(
            "Set MODEL_PATH (local file) or both HF_HUB_REPO + HF_HUB_FILENAME env vars."
        )

    from huggingface_hub import hf_hub_download

    token = os.getenv("HF_TOKEN") or None
    os.makedirs("models", exist_ok=True)
    logger.info("Downloading %s from %s …", filename, repo_id)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        local_dir="models",
    )
    logger.info("Model ready at %s", local_path)
    return local_path


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    from src.rag_pipeline import RAGPipeline

    db_path = os.getenv("DB_PATH", "data/chroma_db")
    model_path = _resolve_model_path()

    logger.info("Initialising RAG pipeline …")
    _pipeline = RAGPipeline(
        db_path=db_path,
        use_llama_cpp=True,
        model_path=model_path,
    )
    logger.info("Pipeline ready.")
    yield
    _pipeline = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NUST Bank RAG API",
    description="Retrieval-Augmented Generation over NUST Bank product knowledge.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------
# Patterns commonly used in prompt-injection / jailbreak attempts
_INJECTION_PATTERNS = re.compile(
    r"ignore (previous|above|all|prior)|forget (the |all |previous |above )"
    r"|you are now|pretend (you are|to be)|act as|jailbreak|DAN mode"
    r"|system prompt|reveal your instructions|repeat after me"
    r"|override (instructions|rules)|new persona",
    re.IGNORECASE,
)

MAX_QUESTION_LENGTH = 500


def _sanitize_question(raw: str) -> str:
    """Strip leading/trailing whitespace and enforce length cap."""
    cleaned = raw.strip()
    # Truncate silently — don't expose the limit in error messages
    if len(cleaned) > MAX_QUESTION_LENGTH:
        cleaned = cleaned[:MAX_QUESTION_LENGTH]
    return cleaned


def _is_injection_attempt(question: str) -> bool:
    """Return True if the question contains known prompt-injection patterns."""
    return bool(_INJECTION_PATTERNS.search(question))


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


class VectorstoreResetResponse(BaseModel):
    status: str
    collection: str


class VectorstoreIngestResponse(BaseModel):
    status: str
    files_processed: int
    chunks_added: int
    warnings: list[str]


# Same word-based chunking as the Streamlit uploader (must match ``process_file`` contract).
class _IngestTokenizer:
    def encode(self, text: str) -> list:
        return text.split()

    def decode(self, tokens: list) -> str:
        return " ".join(tokens)


_INGEST_TOKENIZER = _IngestTokenizer()

ALLOWED_INGEST_EXTENSIONS = frozenset({".pdf", ".docx", ".json", ".txt"})

# Fallback when filename has no extension but Content-Type is known
_CONTENT_TYPE_SUFFIX = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/json": ".json",
    "text/plain": ".txt",
}


def _effective_upload_name(upload: UploadFile, raw: bytes) -> str:
    """Return a basename with an extension we can route (pdf / docx / json / txt)."""
    raw_name = (upload.filename or "").strip() or "upload"
    base_name = os.path.basename(raw_name)
    root, ext = os.path.splitext(base_name)
    ext = ext.lower()
    if ext in ALLOWED_INGEST_EXTENSIONS:
        return base_name

    ct = (upload.content_type or "").split(";")[0].strip().lower()
    if ct in _CONTENT_TYPE_SUFFIX:
        return (root or "upload") + _CONTENT_TYPE_SUFFIX[ct]

    if not ext and ct in ("application/octet-stream", "binary/octet-stream"):
        if raw.startswith(b"%PDF"):
            return (root or "upload") + ".pdf"
        if raw.startswith(b"PK\x03\x04"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    names = zf.namelist()
                if "word/document.xml" in names:
                    return (root or "upload") + ".docx"
            except zipfile.BadZipFile:
                pass
        try:
            json.loads(raw.decode("utf-8"))
            return (root or "upload") + ".json"
        except Exception:
            return (root or "upload") + ".txt"

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported upload {base_name!r} (Content-Type: {ct!r}). "
            "Use .pdf, .docx, .json, or .txt, or send a matching Content-Type."
        ),
    )


MAX_INGEST_BYTES = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/vectorstore/reset", response_model=VectorstoreResetResponse)
def vectorstore_reset():
    """Remove all vectors from the configured Chroma collection and recreate an empty index."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    _pipeline.reset_vectorstore()
    return VectorstoreResetResponse(status="ok", collection=_pipeline.collection_name)


@app.post("/vectorstore/ingest", response_model=VectorstoreIngestResponse)
async def vectorstore_ingest(
    files: Annotated[list[UploadFile], File(description="One or more .pdf, .docx, .json, or .txt files")],
):
    """Parse uploads, normalize text (same pipeline as offline indexing), and append embeddings to Chroma."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not files:
        raise HTTPException(status_code=400, detail="Provide at least one file.")

    from src.data_pipeline import process_upload_bytes

    warnings: list[str] = []
    total_chunks = 0
    processed = 0

    for upload in files:
        raw = await upload.read()
        if len(raw) > MAX_INGEST_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File {upload.filename!r} exceeds limit of {MAX_INGEST_BYTES // (1024 * 1024)} MB.",
            )
        name = _effective_upload_name(upload, raw)
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_INGEST_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Rejected {name!r}: extension must be one of {sorted(ALLOWED_INGEST_EXTENSIONS)}.")

        try:
            chunks = process_upload_bytes(name, raw, _INGEST_TOKENIZER, lowercase=False)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ImportError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        if not chunks:
            warnings.append(f"No extractable chunks from {name!r} (empty or unsupported content).")
            continue

        added = _pipeline.add_chunk_dicts(chunks)
        total_chunks += added
        processed += 1

    return VectorstoreIngestResponse(
        status="ok",
        files_processed=processed,
        chunks_added=total_chunks,
        warnings=warnings,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    question = _sanitize_question(request.question)
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # Reject obvious injection attempts at the API boundary before they reach the LLM
    if _is_injection_attempt(question):
        logger.warning("[security] Injection attempt blocked: %r", question[:120])
        return QueryResponse(
            answer="I can only assist with questions about NUST Bank products and services.",
            sources=[],
        )

    result = _pipeline.query(question)
    return QueryResponse(answer=result["answer"], sources=result["sources"])
