# CUDA 12.1 runtime – pre-built wheel from abetlen index, no nvcc needed
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 + curl only (no cmake/ninja – not compiling from source)
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
# Install llama-cpp-python from the official pre-built CUDA 12.1 wheel index.
# This skips source compilation entirely — install takes ~30 sec instead of 20 min.
# Index: https://abetlen.github.io/llama-cpp-python/whl/cu121
COPY backend/requirements.txt backend/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir llama-cpp-python==0.3.7 \
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
