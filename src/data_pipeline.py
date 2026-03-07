import os
import re
import json
import glob
import argparse
import warnings
import chardet
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
from langdetect import detect

# tokenizer can be provided by your embedding model package (e.g., tiktoken, transformers)

# try to load spaCy; if it fails (Python version issue), disable NER
nlp = None
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")  # for NER masking
except Exception as e:
    print(f"spaCy load failed: {e}. Named-entity masking will be skipped.")

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="confection")

# ---------- Phase 1: Document Loading ----------

def load_txt(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    # encoding detection
    enc = chardet.detect(raw)["encoding"]
    return raw.decode(enc, errors="ignore")


def load_csv(path: str) -> str:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.to_csv(index=False)


def load_excel_rows(path: str) -> list:
    """Read all sheets and return row-level records for better structure."""
    xl = pd.ExcelFile(path)
    records = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, dtype=str).fillna("")
        # Drop columns that are completely empty
        non_empty_cols = [c for c in df.columns if df[c].astype(str).str.strip().any()]
        df = df[non_empty_cols] if non_empty_cols else df
        for idx, row in df.iterrows():
            cells = [str(v).strip() for v in row.values if str(v).strip()]
            if not cells:
                continue
            # Heuristic: if first cell looks like a question and second is present, format as Q/A
            if len(cells) >= 2 and cells[0].rstrip().endswith("?"):
                text = f"Q: {cells[0]}\nA: {cells[1]}"
                if len(cells) > 2:
                    text += "\nNotes: " + " | ".join(cells[2:])
            else:
                text = " | ".join(cells)
            records.append(
                {
                    "text": text,
                    "metadata": {
                        "source": os.path.basename(path),
                        "sheet": sheet,
                        "row_index": int(idx),
                    },
                }
            )
    return records


def load_json_faq(path: str) -> list:
    """Load structured FAQ JSON (categories/questions or list format)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    if isinstance(data, dict) and "categories" in data:
        for cat in data["categories"]:
            category = cat.get("category", "General")
            for q in cat.get("questions", []):
                text = f"Q: {q['question']}\nA: {q['answer']}"
                records.append({
                    "text": text,
                    "metadata": {
                        "source": os.path.basename(path),
                        "category": category,
                    }
                })
    elif isinstance(data, list):
        for i, item in enumerate(data):
            text = json.dumps(item) if isinstance(item, dict) else str(item)
            records.append({
                "text": text,
                "metadata": {"source": os.path.basename(path), "index": i}
            })
    return records


# PDF and DOCX handling would require external packages (pdfminer, python-docx) –
# see comments below for how to integrate

# ---------- Phase 2: Text Cleaning & Normalization ----------

def normalize_encoding(text: str) -> str:
    # assuming 
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def remove_structural_noise(text: str) -> str:
    # strip HTML/XML
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text("\n")


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def standardize_formatting(text: str) -> str:
    # quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # bullets
    text = re.sub(r"^[\u2022\u2023\u25E6\*\-]", "-", text, flags=re.MULTILINE)
    # date formats (example: DD/MM/YYYY -> YYYY-MM-DD)
    text = re.sub(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", lambda m: f"{m.group(3)}-{int(m.group(2)):02}-{int(m.group(1)):02}", text)
    return text


def lowercase_if_needed(text: str, force: bool = False) -> str:
    if force:
        return text.lower()
    return text

# ---------- Phase 3: Data Privacy & Anonymization ----------

regex_patterns = {
    "ACCOUNT_NUM": r"\b\d{6,16}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",
    "PHONE": r"\b\+?\d[\d\- ]{7,}\b",
    "EMAIL": r"[\w\.-]+@[\w\.-]+",
    "CC": r"\b\d{4}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{4}\b",
    "CNIC": r"\b\d{5}-\d{7}-\d\b",
    "NTN": r"\b\d{7}-\d\b|\b\d{7}\b",
    "PASSPORT": r"\b[A-PR-WY][0-9]{7}\b",
    "ACCOUNT_TITLE": r"\baccount title\s*[:\-]\s*[a-z0-9\s\.,&'-]{3,}\b",
    "ADDRESS": r"\b(house|flat|plot|street|road|sector|block)\s*[#\w\-\/]+\b",
}

def mask_sensitive(text: str) -> str:
    for label, pattern in regex_patterns.items():
        text = re.sub(pattern, f"[{label}]", text, flags=re.IGNORECASE)
    # mask generic ID-like sequences (long digit runs) after applying specific patterns
    text = re.sub(r"\b\d{9,}\b", "[ID]", text)
    # mask bank account numbers with separators
    text = re.sub(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,6}\b", "[ACCOUNT_NUM]", text)
    return text


def ner_mask(text: str) -> str:
    if nlp is None:
        # spaCy not available; return original text
        return text
    doc = nlp(text)
    out = []
    for token in doc:
        if token.ent_type_ in {"PERSON", "GPE", "LOC", "ORG"}:
            out.append(f"[{token.ent_type_}]")
        else:
            out.append(token.text_with_ws)
    return "".join(out)

# ---------- Phase 4: Structural Processing ----------

def segment_sections(text: str) -> list:
    # naive heading-based split (look for lines that are all caps or end with ':')
    sections = []
    current = []
    for line in text.splitlines():
        if line.strip().endswith(":") or line.isupper():
            if current:
                sections.append("\n".join(current))
                current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections


def deduplicate(chunks: list) -> list:
    seen = set()
    unique = []
    for chunk in chunks:
        h = hash(chunk)
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique

# ---------- Phase 5: Chunking ----------

def chunk_text(text: str, tokenizer, max_tokens: int = 600, overlap: int = 100) -> list:
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += max_tokens - overlap
    return chunks

# ---------- Phase 6: Metadata Enrichment ----------

def enrich_metadata(chunks: list, **meta) -> list:
    return [{"text": c, "metadata": meta} for c in chunks]

# ---------- Phase 7+: Filtering and final prep ----------

def filter_short(chunks: list, min_tokens: int = 30, tokenizer=None) -> list:
    if tokenizer is None:
        return chunks
    out = []
    for c in chunks:
        if len(tokenizer.encode(c)) >= min_tokens:
            out.append(c)
    return out


def filter_language(chunks: list, lang: str = "en") -> list:
    out = []
    for c in chunks:
        try:
            if detect(c) == lang:
                out.append(c)
        except Exception:
            continue
    return out

def simple_tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b", text)


def sanitize_text(text: str, lowercase: bool) -> str:
    text = normalize_encoding(text)
    text = remove_structural_noise(text)
    text = normalize_whitespace(text)
    text = standardize_formatting(text)
    text = lowercase_if_needed(text, force=lowercase)
    text = mask_sensitive(text)
    text = ner_mask(text)
    return text


def sanitize_records(records: list, lowercase: bool, do_tokenize: bool) -> list:
    out = []
    for rec in records:
        cleaned = sanitize_text(rec["text"], lowercase=lowercase)
        if not cleaned:
            continue
        item = {"text": cleaned, "metadata": rec["metadata"]}
        if do_tokenize:
            item["tokens"] = simple_tokenize(cleaned)
        out.append(item)
    return out

# ---------- Example runner ----------

def process_file(path: str, tokenizer, lowercase: bool = False) -> list:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        text = load_txt(path)
        records = [{"text": text, "metadata": {"source": os.path.basename(path)}}]
    elif ext == ".csv":
        text = load_csv(path)
        records = [{"text": text, "metadata": {"source": os.path.basename(path)}}]
    elif ext == ".json":
        records = load_json_faq(path)
    elif ext in {".xls", ".xlsx"}:
        records = load_excel_rows(path)
    else:
        raise ValueError(f"unsupported format: {ext}")

    records = sanitize_records(records, lowercase=lowercase, do_tokenize=False)

    # Preserve per-record metadata (sheet name, category, etc.)
    output = []
    for rec in records:
        text = rec["text"]
        if not text.strip():
            continue
        # Only chunk texts that are very long
        words = text.split()
        if len(words) > 600:
            chunks = chunk_text(text, tokenizer)
            for chunk in chunks:
                output.append({"text": chunk, "metadata": rec["metadata"]})
        else:
            output.append(rec)

    # Deduplicate by text content
    seen = set()
    unique = []
    for item in output:
        h = hash(item["text"])
        if h not in seen:
            seen.add(h)
            unique.append(item)
    return unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanitize and prepare bank knowledge dataset.")
    parser.add_argument("--input", default=os.getcwd(), help="File or directory to process.")
    parser.add_argument("--output", default="cleaned_chunks.json", help="Output JSON file.")
    parser.add_argument("--lowercase", action="store_true", help="Force lowercase.")
    parser.add_argument("--tokenize", action="store_true", help="Include simple word tokens in output.")
    parser.add_argument("--lang", default="en", help="Language filter (default: en).")
    args = parser.parse_args()

    # placeholder tokenizer using simple whitespace split; replace with a real tokenization
    class DummyTok:
        def encode(self, t): return t.split()
        def decode(self, ts): return " ".join(ts)
    tok = DummyTok()

    outputs = []
    if os.path.isfile(args.input):
        print(f"Processing {args.input}")
        outputs.extend(process_file(args.input, tok, lowercase=args.lowercase))
    else:
        for ext in ("*.txt", "*.csv", "*.json", "*.xls", "*.xlsx"):
            for path in glob.glob(os.path.join(args.input, ext)):
                print(f"Processing {path}")
                outputs.extend(process_file(path, tok, lowercase=args.lowercase))

    # optional tokenization at record-level for downstream workflows
    if args.tokenize:
        outputs = [{"text": o["text"], "metadata": o["metadata"], "tokens": simple_tokenize(o["text"])} for o in outputs]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(outputs)} chunks to {args.output}")
