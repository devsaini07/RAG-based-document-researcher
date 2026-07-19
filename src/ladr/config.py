from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_DIR = PROJECT_ROOT / os.getenv("LADR_DOCS_DIR", "documents")
DEFAULT_VECTORSTORE_DIR = PROJECT_ROOT / os.getenv("LADR_VECTORSTORE_DIR", "vectorstore")
DEFAULT_EMBEDDING_MODEL = os.getenv("LADR_EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_CHAT_MODEL = os.getenv("LADR_CHAT_MODEL", "llama3.2")
DEFAULT_CHUNK_SIZE = int(os.getenv("LADR_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("LADR_CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K = int(os.getenv("LADR_TOP_K", "4"))
COLLECTION_NAME = os.getenv("LADR_COLLECTION_NAME", "ladr_documents")
