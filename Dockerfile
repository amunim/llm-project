# CUDA 12.1 runtime – required for GPU llama-cpp-python on HF Spaces T4
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 + curl for bootstrap pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip into python3.11 specifically (ubuntu's python3-pip targets 3.10)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Make python3.11 the default 'python' and 'python3'
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Install Python deps ---------------------------------------------------------
# 1. Install llama-cpp-python with CUDA 12.1 pre-built wheel BEFORE requirements.txt
#    (so the CUDA wheel is not overwritten by a CPU wheel from PyPI)
COPY backend/requirements.txt backend/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir \
        llama-cpp-python==0.3.7 \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
 && python -m pip install --no-cache-dir -r backend/requirements.txt

# Copy project sources --------------------------------------------------------
COPY . .

# Build ChromaDB vector index at image build time from the committed JSON.
# This avoids committing binary chroma_db files to git.
RUN python src/embeddings.py \
        --input cleaned_chunks.json \
        --db-path data/chroma_db

# HF Spaces runs as non-root user (uid 1000)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
