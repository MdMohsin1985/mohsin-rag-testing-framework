"""Evaluate the RAG app with DeepEval faithfulness and relevancy metrics."""

from __future__ import annotations

import os
import sys
import argparse
import json
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

from rag_app import LATEST_RUN_PATH, answer_question, save_rag_run


ROOT_DIR = Path(__file__).resolve().parents[1]


def require_api_key() -> None:
    """Validate the OpenAI key used by the app and DeepEval metrics."""
    load_dotenv(ROOT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create a .env file from .env.example first."
        )


def build_test_case(question: str, answer: str, context_chunks: list[str]) -> LLMTestCase:
    """Convert one RAG result into a DeepEval test case."""
    return LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context_chunks,
    )


def build_fresh_test_case(question: str, top_k: int) -> tuple[LLMTestCase, str, str, Path]:
    """Run the RAG pipeline now and evaluate that newly generated answer."""
    answer, context_chunks = answer_question(question=question, top_k=top_k)
    run_path = save_rag_run(
        question=question,
        answer=answer,
        context_chunks=context_chunks,
    )
    return (
        build_test_case(
            question=question,
            answer=answer,
            context_chunks=context_chunks,
        ),
        question,
        answer,
        run_path,
    )


def build_saved_test_case(run_path: Path = LATEST_RUN_PATH) -> tuple[LLMTestCase, str, str]:
    """Load the latest saved RAG answer and evaluate that exact prior run."""
    if not run_path.exists():
        raise RuntimeError(
            "No saved RAG run found. Run `python src/rag_app.py --question \"...\"` first "
            "or pass `--question` to evaluate a fresh answer."
        )

    run = json.loads(run_path.read_text(encoding="utf-8"))
    question = run["question"]
    answer = run["answer"]
    return (
        build_test_case(
            question=question,
            answer=answer,
            context_chunks=run["retrieval_context"],
        ),
        question,
        answer,
    )


def run_evaluation(question: str | None = None, top_k: int = 4) -> None:
    """Run Faithfulness and Answer Relevancy on a fresh or saved RAG answer."""
    require_api_key()
    if question:
        print("Evaluation mode: fresh RAG answer")
        test_case, evaluated_question, answer, run_path = build_fresh_test_case(
            question=question,
            top_k=top_k,
        )
        print(f"Saved run: {run_path}")
        print(f"Latest run: {LATEST_RUN_PATH}")
    else:
        print(f"Evaluating saved answer from: {LATEST_RUN_PATH}")
        test_case, evaluated_question, answer = build_saved_test_case()

    print(f"\nQuestion: {evaluated_question}")
    print("\nAnswer:")
    print(f"{answer}\n")

    # DeepEval will use an LLM judge to score groundedness and relevance.
    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
    ]
    evaluate(test_cases=[test_case], metrics=metrics)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Evaluate a fresh RAG answer or the latest saved RAG run."
        )
        parser.add_argument(
            "--question",
            "-q",
            help="Question to evaluate with a fresh RAG run. If omitted, uses rag_runs/latest_run.json.",
        )
        parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve for fresh evaluation.")
        args = parser.parse_args()
        run_evaluation(question=args.question, top_k=args.top_k)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
