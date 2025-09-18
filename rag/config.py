"""
config.py
---------
Shared configuration variables for the RAG pipeline.
Controls file paths, embedding/LLM settings, and retrieval parameters.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ""))
CV_DIR = PROJECT_ROOT / "data" / "samples" / "fake_cvs"
INDEX_CSV = CV_DIR / "index.csv"
PERSIST_DIR = Path(os.getenv("RAG_PERSIST_DIR", ".chroma/cv_rag"))

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM (via OpenRouter, OpenAI-compatible client)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
CHAT_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")

# Retrieval parameters
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
