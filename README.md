# LADR

Local offline document researcher built around Ollama, LangChain, and Chroma.

## What it does

- Ingests documents from the `documents/` folder.
- Splits them into chunks and stores vector embeddings locally in `vectorstore/`.
- Answers questions by retrieving relevant chunks and sending them to a local Ollama model.

## Project layout

- `documents/`: put your source files here.
- `vectorstore/`: persisted local embeddings database.
- `src/ladr/`: reusable ingestion and query logic.
- `scripts/`: thin command wrappers if you want direct script entry points.

## Setup

1. Install and run Ollama locally.
2. Pull the models you want to use, for example:
   - `ollama pull nomic-embed-text`
   - `ollama pull llama3.2`
3. Install Python dependencies:

```bash
pip install -e .
```

## Usage

Ingest documents:

```bash
ladr-ingest
```

Ask a question:

```bash
ladr-ask "What does the document say about retention policy?"
```

Or use the scripts directly:

```bash
python scripts/ingest_documents.py
python scripts/ask_question.py "Your question here"
```

## Notes

- The default embedding model is `nomic-embed-text`.
- The default chat model is `llama3.2`.
- If you want a stronger embedding model and do not mind a larger download, `mxbai-embed-large` is a solid alternative.
- Set `OLLAMA_HOST` if your Ollama server is not on `http://localhost:11434`.
