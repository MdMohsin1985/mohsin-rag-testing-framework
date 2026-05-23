"""Evaluate the RAG app with DeepEval faithfulness and relevancy metrics."""

from __future__ import annotations

import os
import sys
from pathlib import Path


# DeepEval uses Rich output with Unicode characters; make Windows consoles safe.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from rag_app import answer_question


ROOT_DIR = Path(__file__).resolve().parents[1]


def require_api_key() -> None:
    """Validate the OpenAI key used by the app and DeepEval metrics."""
    load_dotenv(ROOT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create a .env file from .env.example first."
        )


def build_test_case(question: str) -> LLMTestCase:
    """Run the RAG pipeline once and convert the result into a DeepEval test case."""
    answer, context_chunks = answer_question(question)
    return LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context_chunks,
    )


def run_evaluation() -> None:
    """Run Faithfulness and Answer Relevancy metrics against a sample question."""
    require_api_key()
    question = "What does the AI RAG Testing Framework evaluate?"
    test_case = build_test_case(question)

    # DeepEval will use an LLM judge to score groundedness and relevance.
    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
    ]
    evaluate(test_cases=[test_case], metrics=metrics)


if __name__ == "__main__":
    try:
        run_evaluation()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
