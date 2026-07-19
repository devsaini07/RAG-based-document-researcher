from __future__ import annotations

import argparse
from pathlib import Path

from ladr.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_DOCS_DIR, DEFAULT_VECTORSTORE_DIR
from ladr.documents import load_documents, split_documents
from ladr.vectorstore import build_vectorstore


def ingest_documents(
    documents_dir: Path | None = None,
    vectorstore_dir: Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str | None = None,
) -> int:
    source_dir = documents_dir or DEFAULT_DOCS_DIR
    target_dir = vectorstore_dir or DEFAULT_VECTORSTORE_DIR
    documents = load_documents(source_dir)
    if not documents:
        return 0

    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore = build_vectorstore(vectorstore_dir=target_dir, model_name=embedding_model)
    vectorstore.add_documents(chunks)
    return len(chunks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest documents into the local vector store.")
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--vectorstore-dir", type=Path, default=DEFAULT_VECTORSTORE_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chunk_count = ingest_documents(
        documents_dir=args.documents_dir,
        vectorstore_dir=args.vectorstore_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )
    print(f"Ingested {chunk_count} chunks from {args.documents_dir} into {args.vectorstore_dir}.")


if __name__ == "__main__":
    main()
