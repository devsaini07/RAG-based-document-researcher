from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from ladr.config import COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL, DEFAULT_VECTORSTORE_DIR


def build_embeddings(model_name: str | None = None) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model_name or DEFAULT_EMBEDDING_MODEL)


def build_vectorstore(vectorstore_dir: Path | None = None, model_name: str | None = None) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(vectorstore_dir or DEFAULT_VECTORSTORE_DIR),
        embedding_function=build_embeddings(model_name),
    )
