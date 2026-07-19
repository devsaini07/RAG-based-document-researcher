from __future__ import annotations

import argparse

from ladr.config import DEFAULT_CHAT_MODEL, DEFAULT_TOP_K, DEFAULT_VECTORSTORE_DIR
from ladr.rag import answer_question


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions against the local document store.")
    parser.add_argument("question", nargs="?", help="Question to ask the researcher")
    parser.add_argument("--vectorstore-dir", default=DEFAULT_VECTORSTORE_DIR)
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    question = args.question or input("Ask a question: ").strip()
    if not question:
        raise SystemExit("A question is required.")

    response = answer_question(
        question=question,
        vectorstore_dir=args.vectorstore_dir,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model or DEFAULT_CHAT_MODEL,
        top_k=args.top_k,
    )
    print("Answer:\n")
    print(response["answer"])
    if response["sources"]:
        print("\nSources:")
        for source in response["sources"]:
            print(f"- {source}")


if __name__ == "__main__":
    main()
