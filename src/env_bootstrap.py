"""
Normalize OpenMP / BLAS thread env vars before native libs (PyTorch, llama.cpp) load.

Hugging Face Spaces sometimes injects OMP_NUM_THREADS="" which triggers:
  libgomp: Invalid value for environment variable OMP_NUM_THREADS
"""

from __future__ import annotations

import os


def ensure_valid_thread_env() -> None:
    """If OMP_NUM_THREADS is missing or invalid, set a sane CPU thread count."""
    raw = os.environ.get("OMP_NUM_THREADS", "").strip()
    ok = raw.isdigit() and int(raw) >= 1
    if not ok:
        n = max(1, min(8, os.cpu_count() or 4))
        os.environ["OMP_NUM_THREADS"] = str(n)
