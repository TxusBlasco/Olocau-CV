====================== START OF README.md ======================

# Olocau-CV

End-to-end prototype that:
1) Generates 25–30 realistic fake CVs in PDF (Task 1).
2) Builds a simple RAG pipeline so an LLM can answer questions grounded on those CVs (Task 2).
3) (Next) A minimal chat UI can be added to query the RAG API.

This project is intentionally local-first and simple to run. You can swap components (vector DB, embedding model, LLM provider) as needed.

Credits: Olocau-CV is an AI-powered app designed by Txus Blasco (https://github.com/TxusBlasco/Olocau-CV)

----------------------------------------------------------------

## Repo Layout

data/
  samples/
    fake_cvs/            # Generated PDFs + index.csv (after Task 1)
tools/
  rag/
    config.py            # Shared configuration for RAG
    ingest_pdfs.py       # Parse PDFs → chunk → embed → Chroma (LangChain modern pkgs)
    query_rag.py         # Retrieve → prompt → LLM answer (CLI)
    api_rag.py           # Optional FastAPI to expose /search
gen_fake_cvs.py          # Task 1: Generate CV PDFs (+ index.csv)
README.md                # ← You are here (single README for entire repo)
requirements.txt         # ← Single consolidated dependencies

Note: The RAG scripts expect CV PDFs at data/samples/fake_cvs/ plus an index.csv (produced by gen_fake_cvs.py).

----------------------------------------------------------------

## Quickstart

0) Python & venv

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

1) Environment Variables

Required (LLM via OpenRouter):
export OPENROUTER_API_KEY=YOUR_KEY
# Optional (defaults shown)
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="mistralai/mistral-7b-instruct:free"

Optional (image headshots, only if CV generator uses Stability):
export STABILITY_API_KEY=YOUR_KEY

Optional knobs for RAG (defaults shown):
export RAG_PERSIST_DIR=".chroma/cv_rag"
export RAG_TOP_K=6
export RAG_CHUNK_SIZE=900
export RAG_CHUNK_OVERLAP=150
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export PROJECT_ROOT="."     # change if you run from a different cwd

Note on telemetry: the RAG scripts programmatically disable LangChain telemetry via:
os.environ["LANGCHAIN_TELEMETRY"] = "false"
so you do not need to export it manually.

2) Generate Fake CVs (Task 1)

python gen_fake_cvs.py --n 28 --seed 13

Output:
- data/samples/fake_cvs/cv_*.pdf
- data/samples/fake_cvs/index.csv

3) Ingest PDFs → Vector Store (Task 2)

cd tools/rag
python ingest_pdfs.py
# Creates/updates a local Chroma collection under $RAG_PERSIST_DIR

4) Ask Questions (CLI)

python query_rag.py --q "Who has Kubernetes or MLOps experience in Valencia?"
python query_rag.py --q "Which candidates speak German at B2 or higher?"
python query_rag.py --q "Find senior Data Engineers available in Madrid."

5) Optional HTTP API

uvicorn api_rag:app --host 0.0.0.0 --port 8000
# Try:
#   http://localhost:8000/search?q=Who has experience with AWS SageMaker?

Example API response (shape):

{
  "answer": "…",
  "sources": [
    {"file":"cv_jane-doe_01.pdf","name":"Jane Doe","role":"Data Scientist","city":"Valencia"}
  ]
}

----------------------------------------------------------------

## How It Works (RAG)

1) Extract text from each cv_*.pdf using pypdf (via LangChain’s PyPDFLoader).
2) Chunk text into ~900-char segments with 150-char overlap (RecursiveCharacterTextSplitter).
3) Embed chunks with a local model (all-MiniLM-L6-v2) via langchain-huggingface (fast & free).
4) Store vectors + metadata in Chroma (langchain-chroma) persisted on disk.
5) On a query, retrieve the top-K most similar chunks and send them as grounding context to the LLM.
6) The LLM is instructed to answer only using these chunks and to cite snippet numbers.

Why Chroma as backend
- Free & local-first (no cloud account or costs).
- Fast iteration: easy to wipe and re-ingest while tuning chunking/prompts.
- Persistent on disk (no rebuild every run).
- Good enough for this dataset size; can swap to Pinecone/Vertex later with minimal code changes via LangChain.

----------------------------------------------------------------

## Hallucinations & Answer Quality

Sometimes the LLM may hallucinate skills or experiences that are semantically related but not explicitly written in the CV (e.g., inferring “Kubernetes” from “Docker” or “cloud”).

Mitigations implemented / recommended:

1) Prompt tightening (implemented in query_rag.py):
   - The prompt enforces: “Use ONLY the information explicitly written in the CV chunks. Do not guess. If not mentioned, answer: ‘Not mentioned.’”
   - This reduces confabulation.

2) Keyword filtering (optional, easy add):
   - Before passing retrieved chunks to the LLM, keep only those containing the queried terms (e.g., “kubernetes”, “mlops”).
   - This raises precision for skills questions.

3) Map-Reduce QA (optional):
   - Use map_reduce chain_type: first ask per-chunk “Does this mention X? Copy the exact sentence.” then aggregate in reduce step.
   - More conservative than stuffing all chunks at once.

4) Post-check guardrail (optional):
   - After the LLM responds, verify that any claimed skill appears verbatim in at least one retrieved chunk. If not, replace with “Not mentioned.”

For this PoC, start with the tightened prompt (already in the code). If you still see issues, add the keyword post-check or switch to map_reduce for skill lookups.

----------------------------------------------------------------

## Configuration

All knobs live in tools/rag/config.py and can be overridden via environment variables:

- Paths: PROJECT_ROOT, CV_DIR, INDEX_CSV, PERSIST_DIR
- Embeddings: EMBEDDING_MODEL
- LLM: OPENROUTER_API_KEY, LLM_BASE_URL, LLM_MODEL
- Retrieval: RAG_TOP_K, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP

----------------------------------------------------------------

## Troubleshooting

- No PDFs found
  Run gen_fake_cvs.py first and confirm files exist in data/samples/fake_cvs/.

- Empty text from PDFs
  Check fonts; try pdfminer.six if needed.

- Embedding slowness
  Batch ingestion or reduce chunk size.

- LLM 401/403
  Verify OPENROUTER_API_KEY.

- Chroma schema errors (e.g., sqlite “no such column: collections.topic”)
  Your .chroma/cv_rag DB was created with a different Chroma version.
  Fix: rm -rf .chroma/cv_rag && re-run ingestion.

- Telemetry warnings (Failed to send telemetry event…)
  Harmless. Telemetry is disabled in code; you can also export: LANGCHAIN_TELEMETRY=false

- Ungrounded answers / hallucinations
  Tighten the prompt and/or apply keyword filtering or the post-check guardrail mentioned above.

====================== END OF README.md ======================
