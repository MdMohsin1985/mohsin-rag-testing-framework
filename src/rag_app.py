"""Command-line RAG app backed by ChromaDB and OpenAI."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT_DIR / "chroma_db"
RUNS_DIR = ROOT_DIR / "rag_runs"
LATEST_RUN_PATH = RUNS_DIR / "latest_run.json"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def require_api_key() -> str:
    """Load and validate the OpenAI API key used by retrieval and generation."""
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create a .env file from .env.example first."
        )
    return api_key


def get_collection():
    """Open the same persistent ChromaDB collection created by ingestion."""
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


def retrieve_context(question: str, top_k: int = 4) -> list[str]:
    """Retrieve the most relevant chunks for a user question."""
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("No chunks found. Run `python src/ingest.py` first.")

    # Chroma uses the same OpenAI embedding model here to embed the question.
    results = collection.query(query_texts=[question], n_results=top_k)
    return results.get("documents", [[]])[0]


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Ask OpenAI to answer using only the retrieved context."""
    api_key = require_api_key()
    client = OpenAI(api_key=api_key)
    context = "\n\n".join(
        f"Context chunk {index + 1}:\n{chunk}"
        for index, chunk in enumerate(context_chunks)
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise RAG assistant. Answer only from the "
                    "provided context. If the context is insufficient, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved context:\n{context}",
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def answer_question(question: str, top_k: int = 4) -> tuple[str, list[str]]:
    """Retrieve context and generate an answer for a question."""
    context_chunks = retrieve_context(question=question, top_k=top_k)
    answer = generate_answer(question=question, context_chunks=context_chunks)
    return answer, context_chunks


def save_rag_run(question: str, answer: str, context_chunks: list[str]) -> Path:
    """Save the exact question, answer, and retrieved context for evaluation."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    run = {
        "created_at": created_at,
        "question": question,
        "answer": answer,
        "retrieval_context": context_chunks,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }

    timestamp = created_at.replace(":", "-")
    run_path = RUNS_DIR / f"rag_run_{timestamp}.json"
    json_text = json.dumps(run, indent=2, ensure_ascii=False)
    run_path.write_text(json_text, encoding="utf-8")
    LATEST_RUN_PATH.write_text(json_text, encoding="utf-8")
    return run_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against your TXT documents.")
    parser.add_argument("--question", "-q", help="Question to ask. If omitted, prompts interactively.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    question = args.question or input("Ask a question: ").strip()
    if not question:
        raise SystemExit("Question cannot be empty.")

    answer, context_chunks = answer_question(question=question, top_k=args.top_k)
    run_path = save_rag_run(
        question=question,
        answer=answer,
        context_chunks=context_chunks,
    )
    print("\nAnswer:")
    print(answer)
    print("\nRetrieved sources:")
    for index, chunk in enumerate(context_chunks, start=1):
        preview = chunk.replace("\n", " ")[:120]
        print(f"{index}. {preview}...")
    print(f"\nSaved run: {run_path}")
    print(f"Latest run: {LATEST_RUN_PATH}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
