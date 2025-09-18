"""
query_rag_lc.py
---------------
LangChain CLI for querying the CV RAG pipeline (modern packages).
"""

import argparse
import os

os.environ["LANGCHAIN_TELEMETRY"] = "false"

from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import (
    PERSIST_DIR, EMBEDDING_MODEL, OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL, CHAT_MODEL, TOP_K
)


TEMPLATE = """You are an assistant that answers HR screening questions strictly grounded on the provided CV chunks. 
Only use the retrieved context to answer. 
If the answer is not present, but you guess it implicitly is, explain the reasons that led you to that answer.
Prefer naming the candidate(s), their role, city, and key experiences when relevant. 
Cite snippet numbers like [1], [2].

Context:
{context}

Question:
{question}

Answer:
"""


def build_chain(top_k: Optional[int] = None):
    """
    Build a RetrievalQA chain over Chroma with HuggingFace embeddings.

    IMPORTANT:
    - RetrievalQA takes input key "query".
    - It forwards that into the prompt as variable "question".
    - Therefore the prompt must declare input_variables ["context", "question"].
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name="cv_chunks",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k or TOP_K})
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


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Query string")
    ap.add_argument("--k", type=int, default=TOP_K, help="Top-k results")
    args = ap.parse_args()

    chain = build_chain(top_k=args.k)
    res = chain.invoke({"query": args.q})

    print("---- ANSWER ----")
    print(res["result"])
    if res.get("source_documents"):
        print("\n---- SOURCES ----")
        for i, doc in enumerate(res["source_documents"], 1):
            print(f"[{i}] {doc.metadata.get('file')}")


if __name__ == "__main__":
    main()
