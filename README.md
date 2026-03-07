# NUST Bank Customer Service Assistant

An LLM-powered Retrieval-Augmented Generation (RAG) system that answers customer queries about NUST Bank's products and services using a knowledge base of 35+ product modules.

**Model:** Qwen2.5-3B-Instruct (3.09B parameters, within the 6B course limit)
**Stack:** LangChain · ChromaDB · Sentence-Transformers · Streamlit · UnSloth (QLoRA fine-tuning)

## Architecture

The full architecture diagram is available as a Mermaid file at [`docs/architecture.mmd`](docs/architecture.mmd). Render it at [mermaid.live](https://mermaid.live) or any Mermaid-compatible viewer.

```
User Query
    │
    ▼
┌──────────────┐     ┌───────────────────┐
│  Streamlit   │────▶│  LangChain        │
│  Chat UI     │     │  RetrievalQA      │
└──────────────┘     └───────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌─────────────┐ ┌────────────┐ ┌────────────────┐
     │ ChromaDB    │ │ Qwen2.5-3B │ │ System Prompt  │
     │ Vector      │ │ Instruct   │ │ & Guard Rails  │
     │ Retrieval   │ │ (4-bit)    │ │                │
     └──────┬──────┘ └─────┬──────┘ └────────────────┘
            │              │
     ┌──────┴──────┐       │  (optional)
     │ all-MiniLM  │  ┌────┴─────┐
     │ -L6-v2      │  │ QLoRA    │
     │ Embeddings  │  │ LoRA     │
     └──────┬──────┘  │ Adapter  │
            │         └──────────┘
     ┌──────┴──────┐
     │ Preprocessed│
     │ Chunks      │
     │ (303 items) │
     └─────────────┘
```

## Data Pipeline

The data pipeline processes raw bank documents into semantically coherent chunks:

1. **Ingestion** — Reads Excel (36 sheets, 35+ products) and JSON FAQ files
2. **Row grouping** — Adjacent Excel rows are merged into Q&A sections (each question row starts a new section, subsequent rows form its answer)
3. **Formatting** — Decimal percentages converted to display format (e.g. `0.19` → `19.00%`)
4. **Cleaning** — Encoding normalization, HTML stripping, whitespace cleanup, smart quote standardization
5. **PII anonymization** — Regex-based masking of account numbers, IBANs, CNICs, phone numbers, emails, etc. Optional spaCy NER masking for names/locations
6. **Deduplication** — Hash-based removal of duplicate chunks

**Output:** 303 semantically complete chunks (down from 936 raw rows) with preserved sheet/category metadata.

## Repository Structure
```
├── configs/pipeline.yaml             # Full pipeline configuration
├── data/
│   ├── raw/                          # Source datasets
│   │   ├── NUST Bank-Product-Knowledge.xlsx  (36 sheets)
│   │   └── funds_transfer_app_features_faq.json
│   ├── processed/cleaned_chunks.json # Pipeline output (303 chunks)
│   └── chroma_db/                    # ChromaDB persistent vector store
├── docs/
│   ├── PROJECT.md                    # Project tracking
│   └── architecture.mmd             # Mermaid architecture diagram
├── notebooks/
│   ├── fine_tuning.ipynb             # UnSloth QLoRA fine-tuning (Colab T4)
│   └── rag_inference.ipynb           # Full RAG demo (Colab T4)
├── scripts/run_pipeline.ps1
├── src/
│   ├── data_pipeline.py              # Data ingestion, cleaning & PII masking
│   ├── embeddings.py                 # Sentence-Transformer → ChromaDB indexing
│   ├── rag_pipeline.py               # LangChain RAG chain with guard rails
│   └── app.py                        # Streamlit chat UI
└── requirements.txt
```

## Quick Start

```powershell
# 1. Setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run data preprocessing (produces 303 chunks)
python src/data_pipeline.py --input data/raw --output data/processed/cleaned_chunks.json --tokenize

# 3. Build vector index (embeds & indexes into ChromaDB)
python src/embeddings.py

# 4. Launch the assistant (requires GPU for LLM inference)
streamlit run src/app.py
```

## Colab Notebooks

Local GPU (4GB VRAM) is insufficient for Qwen2.5-3B. Both notebooks are designed for **Google Colab with T4 GPU** (free tier, 16GB VRAM).

### Fine-Tuning (`notebooks/fine_tuning.ipynb`)
- Upload `cleaned_chunks.json` → auto-converts to instruction format (301 examples)
- Loads Qwen2.5-3B via UnSloth in 4-bit quantization
- QLoRA: rank 16, targets q/k/v/o/gate/up/down projections (29.9M trainable params, 0.96%)
- Trains for 3 epochs (57 steps), final loss: ~1.85
- Saves LoRA adapter for download

### RAG Inference (`notebooks/rag_inference.ipynb`)
- Upload `cleaned_chunks.json` → builds in-memory ChromaDB index
- Loads Qwen2.5-3B with BitsAndBytes 4-bit (~2GB VRAM)
- Full RAG query function: retrieve top-5 → build system prompt → generate
- Includes test queries, interactive chat loop, and guard rails tests

## Guard Rails

The system prompt enforces:
- **Domain restriction** — Only answers NUST Bank product/service questions
- **No financial advice** — Refuses investment/savings recommendations
- **Helpline fallback** — Redirects to +92 (51) 111 000 494 when info is unavailable
- **Injection resistance** — Deflects prompt injection and jailbreak attempts
- **Privacy** — Refuses to disclose customer account details

## Notes
- `spacy` is optional — if installed, NER-based PII masking (names, locations) runs alongside regex masking
- The Streamlit UI supports uploading new documents for real-time knowledge base updates
- The fine-tuned LoRA adapter can optionally be loaded into the RAG pipeline for improved response quality
