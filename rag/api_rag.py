from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict

from rag_utils import build_chain

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict]

app = FastAPI(title="CV RAG API (LangChain modern)", version="0.1.0")
chain = build_chain()

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/search", response_model=RAGResponse)
def search(q: str = Query(..., description="Your question")):
    res = chain.invoke({"query": q})
    src_docs = res.get("source_documents", [])
    sources = [{"file": d.metadata.get("file"), "path": d.metadata.get("path")} for d in src_docs]
    return RAGResponse(answer=res["result"], sources=sources)
