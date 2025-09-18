"""
ingest_pdfs_lc.py
-----------------
LangChain ingestion for the CV RAG pipeline (modern packages).

Steps:
1) Load PDFs with PyPDFLoader.
2) Split text into chunks with RecursiveCharacterTextSplitter.
3) Embed with HuggingFaceEmbeddings.
4) Store in a persistent Chroma vector store.
"""

import os

os.environ["LANGCHAIN_TELEMETRY"] = "false"

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CV_DIR, PERSIST_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def find_pdfs(cv_dir: Path) -> List[Path]:
    """Return a sorted list of CV PDFs in the given directory."""
    return sorted(cv_dir.glob("cv_*.pdf"))


def load_pdf_documents(pdf_paths: List[Path]):
    """Load PDFs into LangChain Document objects."""
    docs = []
    for p in pdf_paths:
        loader = PyPDFLoader(str(p))
        page_docs = loader.load()
        for d in page_docs:
            d.metadata = {**d.metadata, "file": p.name, "path": str(p)}
        docs.extend(page_docs)
    return docs


def split_documents(documents):
    """Split Documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    """Create/update Chroma vector store with given chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    vectordb = Chroma(
        collection_name="cv_chunks",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    vectordb.add_documents(chunks)
    return vectordb


def main():
    """Main entry point for ingestion."""
    CV_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = find_pdfs(CV_DIR)
    if not pdfs:
        print(f"[error] No PDFs found in {CV_DIR}. Run gen_fake_cvs.py first.")
        return

    print(f"[info] Loading {len(pdfs)} PDFs ...")
    docs = load_pdf_documents(pdfs)

    print(f"[info] Splitting {len(docs)} documents into chunks ...")
    chunks = split_documents(docs)

    print("[info] Building/updating Chroma vector store ...")
    _ = build_vectorstore(chunks)

    print(f"[ok] Ingested {len(chunks)} chunks into Chroma at {PERSIST_DIR}")


if __name__ == "__main__":
    main()
