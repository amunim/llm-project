# LLM-Based Bank Support Assistant

**Introduction**
This project designs a Large Language Model (LLM) solution to enhance customer service for a local bank. The goal is to transform a curated, anonymized set of customer interaction documents into a responsive AI assistant that can answer customer questions accurately, produce coherent and context-aware responses, and maintain a high standard of data privacy and trust.

**Dataset Description**
You will be provided a dataset from a fictional bank. Your responsibility is to parse and preprocess the dataset, which may include JSON, CSV, or plain-text formats. The dataset is provided via the LMS and is used to build sanitized, LLM-ready training or retrieval data.

**Model Choice (<= 6B Parameters)**
Chosen model: **Gemma 3 1B**.

Why this fits the requirements:
- Open-source and allowed for non-commercial/academic use
- Within the 6B parameter limit
- Small footprint for local development and iteration
- Suitable for prompt engineering and lightweight fine-tuning (LoRA/QLoRA)

**Repository Structure**
```
llm-project/
  README.md
  requirements.txt
  configs/
    pipeline.yaml
  data/
    raw/
      NUST Bank-Product-Knowledge.xlsx
    processed/
      cleaned_chunks.json
  docs/
    PROJECT.md
  scripts/
    run_pipeline.ps1
  src/
    data_pipeline.py
  tests/
```

**Quick Start**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run preprocessing
.\scripts\run_pipeline.ps1
```

Notes:
- `spacy` is optional. If installed and a compatible model is available, NER masking will run.
- Output is written to `data/processed/cleaned_chunks.json`.
