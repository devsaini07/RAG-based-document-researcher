from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_ollama import ChatOllama

from ladr.config import DEFAULT_CHAT_MODEL, DEFAULT_TOP_K
from ladr.vectorstore import build_vectorstore

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a local document researcher. Answer only from the retrieved context. "
            "If the answer is not in the context, say you do not know and suggest what document to inspect.",
        ),
        ("human", "Question: {input}\n\nContext:\n{context}"),
    ]
)


def build_chat_model(model_name: str | None = None) -> ChatOllama:
    return ChatOllama(model=model_name or DEFAULT_CHAT_MODEL)


def build_retriever(vectorstore_dir: Path | None = None, embedding_model: str | None = None, top_k: int = DEFAULT_TOP_K):
    vectorstore: Chroma = build_vectorstore(vectorstore_dir=vectorstore_dir, model_name=embedding_model)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def answer_question(
    question: str,
    vectorstore_dir: Path | None = None,
    embedding_model: str | None = None,
    chat_model: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, object]:
    retriever = build_retriever(vectorstore_dir=vectorstore_dir, embedding_model=embedding_model, top_k=top_k)
    model = build_chat_model(chat_model)
    combine_docs_chain = create_stuff_documents_chain(model, PROMPT)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    result = retrieval_chain.invoke({"input": question})
    answer = result.get("answer", "")
    context_documents = result.get("context", [])
    sources = []
    for document in context_documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")
        if page is not None:
            sources.append(f"{source}#page={page + 1}")
        else:
            sources.append(source)
    return {"answer": answer, "sources": sources}
