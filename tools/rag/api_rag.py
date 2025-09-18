"""
api_rag_lc.py
-------------
FastAPI service exposing the LangChain RAG pipeline (modern packages).
"""

import os

os.environ["LANGCHAIN_TELEMETRY"] = "false"

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import (
    PERSIST_DIR, EMBEDDING_MODEL, OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL, CHAT_MODEL, TOP_K
)

SYSTEM_INSTRUCTIONS = (
    "You are an assistant that answers HR screening questions strictly grounded on the provided CV chunks. "
    "Only use the retrieved context to answer. If the answer is not present, say you don't know."
)

PROMPT_TEMPLATE = """{system_instructions}

Context:
{context}

Question:
{question}

Answer:
"""


def build_chain():
    """Build a RetrievalQA chain for the API process."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name="cv_chunks",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=CHAT_MODEL,
        temperature=0.2,
    )
    prompt = PromptTemplate(
        input_variables=["system_instructions", "context", "question"],
        template=PROMPT_TEMPLATE,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
        return_source_documents=True,
    )


class RAGResponse(BaseModel):
    """Response schema for RAG API."""
    answer: str
    sources: List[Dict]


app = FastAPI(title="CV RAG API (LangChain modern)", version="0.1.0")
chain = build_chain()


@app.get("/healthz")
def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/search", response_model=RAGResponse)
def search(q: str = Query(..., description="Your question")):
    """Main RAG endpoint."""
    res = chain.invoke({"system_instructions": SYSTEM_INSTRUCTIONS, "query": q})
    src_docs = res.get("source_documents", [])
    sources = [{"file": d.metadata.get("file"), "path": d.metadata.get("path")} for d in src_docs]
    return RAGResponse(answer=res["result"], sources=sources)
