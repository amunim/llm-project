"""
FastAPI backend for the NUST Bank RAG system.

Environment variables:
  MODEL_PATH       - local path to the GGUF file (highest priority)
  HF_HUB_REPO      - HuggingFace Hub repo id  (e.g. zain-0/nust-bank-qwen2.5-3b-gguf)
  HF_HUB_FILENAME  - filename inside the repo  (e.g. nust_bank_qwen2.5_3b_q4km.gguf)
  HF_TOKEN         - HuggingFace read token (optional for public repos)
  DB_PATH          - override ChromaDB location (default: data/chroma_db)

Local test:
  $env:MODEL_PATH = "models/nust_bank_qwen2.5_3b_q4km.gguf"
  uvicorn backend.main:app --reload --port 8000
  # POST http://localhost:8000/query  {"question": "What is the Little Champs Account?"}
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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
# Request / Response schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    result = _pipeline.query(request.question)
    return QueryResponse(answer=result["answer"], sources=result["sources"])
