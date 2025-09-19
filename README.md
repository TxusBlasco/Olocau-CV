# Olocau-CV

End-to-end prototype that:  
1. Generates 25–30 realistic fake CVs in PDF (Task 1).  
2. Builds a simple RAG pipeline so an LLM can answer questions grounded on those CVs (Task 2).  
3. Provides a minimal chat UI to query the RAG API (Task 3).

This project is intentionally **local-first** and simple to run. You can swap components (vector DB, embedding model, LLM provider) as needed.

Credits: Olocau-CV is an AI-powered app designed by Txus Blasco  
Repo: https://github.com/TxusBlasco/Olocau-CV

---

## Repo Layout

data/  
-- samples/  
---- fake_cvs/           # Generated PDFs + index.csv (after Task 1)

rag/  
-- config.py             # Shared configuration  
-- rag_utils.py          # Build RAG chain  
-- ingest_pdfs.py        # Parse PDFs → chunk → embed → Chroma- query_rag.py          # Retrieve → prompt → LLM answer (CLI)  
-- api_rag.py            # FastAPI backend exposing /search  

frontend/  
-- frontend_chat.py      # Streamlit chat interface  

tools/  
-- gen_fake_cvs.py       # Task 1: Generate CV PDFs (+ index.csv)  

README.md  
requirements.txt  

Note: the RAG scripts expect CV PDFs at `data/samples/fake_cvs/` plus an `index.csv` (produced by `gen_fake_cvs.py`).

---

## Quickstart

### 0. Python & venv

python -m venv .venv  
source .venv/bin/activate  
pip install --upgrade pip  
pip install -r requirements.txt  

### 1. Environment Variables

**Required (LLM via OpenRouter):**  
export OPENROUTER_API_KEY=YOUR_KEY  

**Optional (defaults shown)**  
export LLM_BASE_URL="https://openrouter.ai/api/v1"  
export LLM_MODEL="mistralai/mistral-7b-instruct:free"  

**Optional (image headshots for CV generation):**  
export STABILITY_API_KEY=YOUR_KEY  

**Optional knobs for RAG (defaults shown):**  
export RAG_TOP_K=12  
export RAG_CHUNK_SIZE=500  
export RAG_CHUNK_OVERLAP=200  
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"  

Note: LangChain telemetry is programmatically disabled in code (`LANGCHAIN_TELEMETRY=false`), no need to export manually.

---

### 2. Generate Fake CVs (Task 1)

python tools/gen_fake_cvs.py --n 28 --seed 13  

Output:  
- data/samples/fake_cvs/cv_*.pdf  
- data/samples/fake_cvs/index.csv  

---

### 3. Ingest PDFs → Vector Store (Task 2)

python rag/ingest_pdfs.py  

Creates/updates a local Chroma collection under `.chroma/cv_rag`.

---

### 4. Ask Questions (CLI)

python rag/query_rag.py --q "Who has Kubernetes or MLOps experience in Valencia?"  
python rag/query_rag.py --q "Which candidates speak German at B2 or higher?"  
python rag/query_rag.py --q "Find senior Data Engineers available in Madrid."  

---

### 5. Run the HTTP API

From rag folder, run:
uvicorn api_rag:app --host 0.0.0.0 --port 8000  

Example:  
curl "http://localhost:8000/search?q=Who has experience with AWS SageMaker?"  

---

### 6. Run the Frontend (Task 3)

streamlit run frontend/frontend_chat.py  

Then open: http://localhost:8501  

---

## How It Works (RAG)

1. Extract text from each `cv_*.pdf` using LangChain’s `PyPDFLoader`.  
2. Split into ~500-char segments with 200 overlap.  
3. Embed with HuggingFace `all-MiniLM-L6-v2`.  
4. Store in Chroma (persisted on disk).  
5. On query, retrieve top-K similar chunks and pass as context to LLM.  
6. The LLM answers **only using these chunks** and cites sources.

---

## Hallucinations & Answer Quality

Mitigations applied:  
- Prompt enforces: “Use ONLY explicit info. If not found, answer: Not found in provided CVs.”  
- Conservative retrieval settings (`mmr` search).  
- Sources are explicitly returned with answers.

Optional improvements:  
- Keyword filtering before LLM.  
- Map-reduce chain.  
- Post-check guardrails.

---

## Configuration

All knobs live in `rag/config.py` and can be overridden by env vars:

- **Paths**: PROJECT_ROOT, CV_DIR, INDEX_CSV, PERSIST_DIR  
- **Embeddings**: EMBEDDING_MODEL  
- **LLM**: OPENROUTER_API_KEY, LLM_BASE_URL, LLM_MODEL  
- **Retrieval**: RAG_TOP_K, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP  

---

## Troubleshooting

- **No PDFs found**  
  Run `tools/gen_fake_cvs.py` first.

- **Empty text from PDFs**  
  Check fonts; try `pdfminer.six` if needed.

- **Chroma schema errors**  
  Wipe and rebuild:  
  rm -rf .chroma/cv_rag  
  python rag/ingest_pdfs.py  

- **LLM 401/403**  
  Check `OPENROUTER_API_KEY`.

- **Ungrounded answers**  
  Tighten prompt or apply keyword filtering.

---
