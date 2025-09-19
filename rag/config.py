"""
config.py
---------
Shared configuration variables for the RAG pipeline.
Controls file paths, embedding/LLM settings, and retrieval parameters.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CV_DIR = PROJECT_ROOT / "data" / "samples" / "fake_cvs"
CV_DIR.mkdir(parents=True, exist_ok=True)
INDEX_CSV = CV_DIR / "index.csv"
PERSIST_DIR = PROJECT_ROOT / ".chroma" / "cv_rag"

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Backend selection: "google" (default) or "openrouter"
LLM_BACKEND = os.getenv("LLM_BACKEND", "google")

# Google AI Studio
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

# LLM (via OpenRouter, OpenAI-compatible client) --> Deprecated (reached the rate limit)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
CHAT_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-8b-instruct:free")

# Retrieval parameters
TOP_K = int(os.getenv("RAG_TOP_K", "12"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
