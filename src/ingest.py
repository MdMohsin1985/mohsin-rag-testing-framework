"""Ingest text documents into a persistent ChromaDB collection."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT_DIR / "documents"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "text-embedding-3-small"


def require_api_key() -> str:
    """Load and validate the OpenAI API key used for embeddings."""
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create a .env file from .env.example first."
        )
    return api_key


def read_text_files(documents_dir: Path = DOCUMENTS_DIR) -> Iterable[tuple[Path, str]]:
    """Yield every non-empty .txt document in the documents folder."""
    documents_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(documents_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            yield path, text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_collection():
    """Create or load the ChromaDB collection with OpenAI embeddings."""
    api_key = require_api_key()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_function = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def build_chunk_id(path: Path, index: int, chunk: str) -> str:
    """Build a stable ID so re-running ingestion updates existing chunks."""
    digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}-{index}-{digest}"


def ingest_documents() -> int:
    """Read TXT files, split them, and upsert the chunks into ChromaDB."""
    collection = get_collection()
    ids: list[str] = []
    chunks: list[str] = []
    metadatas: list[dict[str, str | int]] = []

    for path, text in read_text_files():
        for index, chunk in enumerate(chunk_text(text)):
            ids.append(build_chunk_id(path, index, chunk))
            chunks.append(chunk)
            metadatas.append({"source": path.name, "chunk_index": index})

    if not chunks:
        print(f"No .txt documents found in {DOCUMENTS_DIR}")
        return 0

    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    print(f"Ingested {len(chunks)} chunks from {len(set(m['source'] for m in metadatas))} files.")
    return len(chunks)


if __name__ == "__main__":
    try:
        ingest_documents()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
