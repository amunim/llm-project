# NUST Bank Customer Service Assistant

An LLM-powered Retrieval-Augmented Generation (RAG) system that answers customer queries using NUST Bank's product knowledge base (35+ product modules).

**Model:** Qwen2.5-3B-Instruct (3.09B parameters, within the 6B limit)
**Stack:** LangChain · ChromaDB · Sentence-Transformers · Streamlit · UnSloth (QLoRA)

## Architecture
```
User Query
    │
    ▼
┌──────────────┐     ┌───────────────────┐
│  Streamlit   │────▶│  LangChain RAG    │
│  Chat UI     │     │  Pipeline         │
└──────────────┘     └───────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌─────────────┐ ┌────────────┐ ┌────────────────┐
     │ ChromaDB    │ │ Qwen2.5-3B │ │ Prompt         │
     │ Vector      │ │ Instruct   │ │ Engineering    │
     │ Retrieval   │ │ (LLM)      │ │ (Guard Rails)  │
     └──────┬──────┘ └────────────┘ └────────────────┘
            │
     ┌──────┴──────┐
     │ Embeddings  │
     │ (MiniLM)    │
     └──────┬──────┘
            │
     ┌──────┴──────┐
     │ Preprocessed│
     │ Chunks      │
     │ (936 items) │
     └─────────────┘
```

## Repository Structure
```
├── configs/pipeline.yaml
├── data/
│   ├── raw/                          # Source datasets
│   │   ├── NUST Bank-Product-Knowledge.xlsx
│   │   └── funds_transfer_app_features_faq.json
│   ├── processed/cleaned_chunks.json # Pipeline output
│   └── chroma_db/                    # Vector store
├── docs/PROJECT.md
├── notebooks/
│   └── fine_tuning.ipynb             # UnSloth QLoRA (Colab)
├── scripts/run_pipeline.ps1
├── src/
│   ├── data_pipeline.py              # Data ingestion & PII anonymization
│   ├── embeddings.py                 # Sentence-Transformer → ChromaDB
│   ├── rag_pipeline.py               # LangChain RAG chain
│   └── app.py                        # Streamlit chat UI
└── requirements.txt
```

## Quick Start

```powershell
# 1. Setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run data preprocessing
python src/data_pipeline.py --input data/raw --output data/processed/cleaned_chunks.json --tokenize

# 3. Build vector index
python src/embeddings.py

# 4. Launch the assistant
streamlit run src/app.py
```

## Fine-Tuning (Colab)
Open `notebooks/fine_tuning.ipynb` in Google Colab (T4 GPU). Upload `data/processed/cleaned_chunks.json` and run all cells. The notebook uses UnSloth with QLoRA (4-bit) to fine-tune Qwen2.5-3B-Instruct on the bank dataset.

## Notes
- `spacy` is optional — if installed, NER-based PII masking (names, locations) runs alongside regex masking.
- The Streamlit UI supports uploading new documents for real-time knowledge base updates.
- GPU is recommended for LLM inference. CPU works but will be slower.
