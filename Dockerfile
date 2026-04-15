# CUDA 12.1 devel – nvcc needed to compile llama-cpp-python with CUDA support
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 + build tools for compiling llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    build-essential \
    cmake \
    ninja-build \
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
# Compile llama-cpp-python from source with CUDA 12.1 support.
# Once the wheel is cached in zain-0/llm-wheels (see notebooks/build_llama_wheel.ipynb),
# switch back to the runtime image and install from the cached URL instead.
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

COPY backend/requirements.txt backend/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir llama-cpp-python==0.3.7 \
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
