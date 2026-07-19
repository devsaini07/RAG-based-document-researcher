from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def discover_document_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    return [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def load_documents(source_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in discover_document_files(source_dir):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            loaded_documents = loader.load()
        else:
            loader = TextLoader(str(path), encoding="utf-8")
            loaded_documents = loader.load()

        for document in loaded_documents:
            document.metadata["source"] = str(path)
            documents.append(document)
    return documents


def split_documents(documents: Iterable[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(list(documents))
