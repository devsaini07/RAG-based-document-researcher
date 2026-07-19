from __future__ import annotations

import argparse
import sys
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ladr.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DOCS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_VECTORSTORE_DIR,
)
from ladr.documents import discover_document_files
from ladr.ingest import ingest_documents
from ladr.rag import answer_question

DEFAULT_QUESTIONS = [
    "Summarize the document in 5 bullets.",
    "What are the main topics, skills, or themes mentioned in the document?",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuously smoke-test the offline RAG project.")
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--vectorstore-dir", type=Path, default=DEFAULT_VECTORSTORE_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--interval", type=int, default=0, help="Seconds between repeated test runs. 0 runs once.")
    parser.add_argument("--question", action="append", dest="questions", help="Additional smoke test question.")
    return parser


def reset_vectorstore(vectorstore_dir: Path) -> None:
    if vectorstore_dir.exists():
        shutil.rmtree(vectorstore_dir)


def run_smoke_test(args: argparse.Namespace) -> bool:
    documents = discover_document_files(args.documents_dir)
    print(f"Found {len(documents)} document(s) in {args.documents_dir}.")
    for document in documents:
        print(f"- {document.name}")

    if not documents:
        print("No documents found. Add a PDF or text file to documents/ and run again.")
        return False

    reset_vectorstore(args.vectorstore_dir)
    chunk_count = ingest_documents(
        documents_dir=args.documents_dir,
        vectorstore_dir=args.vectorstore_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )
    print(f"Rebuilt vector store with {chunk_count} chunk(s).")

    questions = DEFAULT_QUESTIONS + list(args.questions or [])
    for question in questions:
        print("\nQuestion:")
        print(question)
        response = answer_question(
            question=question,
            vectorstore_dir=args.vectorstore_dir,
            embedding_model=args.embedding_model,
            chat_model=args.chat_model,
            top_k=args.top_k,
        )
        print("Answer:")
        print(response["answer"])
        if response["sources"]:
            print("Sources:")
            for source in response["sources"]:
                print(f"- {source}")

    return True


def main() -> None:
    args = build_parser().parse_args()
    while True:
        success = run_smoke_test(args)
        if args.interval <= 0:
            raise SystemExit(0 if success else 1)
        print(f"\nSleeping {args.interval} seconds before the next test run...\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()