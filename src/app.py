"""
NUST Bank Customer Service Assistant – Streamlit UI.

Run:
    streamlit run src/app.py
"""

import os
import sys
import json
import streamlit as st

# Ensure src/ is on the path so imports work when launched from project root
sys.path.insert(0, os.path.dirname(__file__))

from rag_pipeline import RAGPipeline
from embeddings import build_index
from data_pipeline import process_file

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="NUST Bank Assistant", page_icon="🏦", layout="centered")
st.title("🏦 NUST Bank Customer Service Assistant")
st.caption("Ask anything about NUST Bank products & services.")

# ── Sidebar: document upload for real-time updates ────────────
with st.sidebar:
    st.header("📄 Upload New Documents")
    st.write("Add new bank documents (Excel, CSV, JSON, or TXT) to update the knowledge base.")
    uploaded = st.file_uploader(
        "Choose file(s)", type=["xlsx", "xls", "csv", "json", "txt"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Index uploaded documents"):
        upload_dir = os.path.join("data", "raw")
        os.makedirs(upload_dir, exist_ok=True)

        class _Tok:
            def encode(self, t): return t.split()
            def decode(self, ts): return " ".join(ts)

        tok = _Tok()
        new_chunks = []
        for f in uploaded:
            dest = os.path.join(upload_dir, f.name)
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            new_chunks.extend(process_file(dest, tok, lowercase=False))

        if new_chunks:
            # Append to existing chunks
            chunks_path = os.path.join("data", "processed", "cleaned_chunks.json")
            existing = []
            if os.path.exists(chunks_path):
                with open(chunks_path, "r", encoding="utf-8") as fj:
                    existing = json.load(fj)
            existing.extend(new_chunks)
            os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
            with open(chunks_path, "w", encoding="utf-8") as fj:
                json.dump(existing, fj, ensure_ascii=False, indent=2)

            # Rebuild index
            build_index(chunks_path, "data/chroma_db")
            # Clear cached pipeline so it reloads
            st.cache_resource.clear()
            st.success(f"Indexed {len(new_chunks)} new chunks from {len(uploaded)} file(s).")
        else:
            st.warning("No processable content found in uploaded files.")

# ── Load RAG pipeline (cached) ───────────────────────────────
@st.cache_resource(show_spinner="Loading model & vector store …")
def get_pipeline():
    return RAGPipeline()

pipeline = get_pipeline()

# ── Chat history ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── User input ───────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about NUST Bank…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            result = pipeline.query(prompt)
            answer = result["answer"]
        st.markdown(answer)

        # Show sources in an expander
        if result.get("sources"):
            with st.expander("📚 Sources"):
                for src in result["sources"]:
                    src_label = src.get("sheet") or src.get("category") or src.get("source", "")
                    st.markdown(f"**{src_label}**: {src.get('text', '')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
