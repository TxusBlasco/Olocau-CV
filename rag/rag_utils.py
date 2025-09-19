"""
rag_utils.py
------------
Shared utilities for building the RAG pipeline.
"""

import os
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    PERSIST_DIR, EMBEDDING_MODEL, TOP_K,
    LLM_BACKEND, GOOGLE_API_KEY, GOOGLE_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, CHAT_MODEL
)

os.environ["LANGCHAIN_TELEMETRY"] = "false"

# Shared prompt
TEMPLATE = """You are an assistant that answers HR screening questions about candidates' CVs. 
You MUST only use the retrieved CV context. 
- Do not speculate or invent facts not explicitly present. 
- If the answer cannot be found, reply "Not found in provided CVs".
- Always list candidate names, city, role, and experiences when relevant.
- Cite sources like [1], [2].


Context:
{context}

Question:
{question}

Answer:
"""

def build_chain(top_k: Optional[int] = None):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name="cv_chunks",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k or TOP_K}
    )

    import os
    print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))

    # Select LLM backend
    if LLM_BACKEND == "google":
        llm = ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
        )
    else:  # openrouter
        llm = ChatOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            model=CHAT_MODEL,
            temperature=0.2,
        )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=TEMPLATE,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
        return_source_documents=True,
    )
