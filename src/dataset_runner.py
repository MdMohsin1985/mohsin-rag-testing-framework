"""Run dataset-based RAG evaluations and write a CSV report."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from evaluate import require_api_key
from rag_app import generate_answer, get_collection


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "datasets" / "evaluation_dataset.json"
REPORTS_DIR = ROOT_DIR / "reports"
REPORT_PATH = REPORTS_DIR / "evaluation_report.csv"
DEFAULT_SOURCE = "sample.txt"


@dataclass(frozen=True)
class DatasetCase:
    """One question and reference answer from the evaluation dataset."""

    question: str
    expected_answer: str


@dataclass(frozen=True)
class EvaluationResult:
    """Serializable result for one evaluated dataset case."""

    case_number: int
    question: str
    expected_answer: str
    actual_answer: str
    retrieval_context: list[str]
    faithfulness_score: float | None
    faithfulness_success: bool | None
    faithfulness_reason: str
    answer_relevance_score: float | None
    answer_relevance_success: bool | None
    answer_relevance_reason: str
    correctness_score: float | None
    correctness_success: bool | None
    correctness_reason: str
    evaluated_at: str


def load_dataset(dataset_path: Path = DATASET_PATH) -> list[DatasetCase]:
    """Read question and expected-answer pairs from the dataset JSON file."""
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset not found: {dataset_path}")

    raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise RuntimeError("Dataset must be a JSON array of test cases.")

    cases: list[DatasetCase] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise RuntimeError(f"Dataset case {index} must be a JSON object.")

        question = str(raw_case.get("question", "")).strip()
        expected_answer = str(raw_case.get("expected_answer", "")).strip()
        if not question or not expected_answer:
            raise RuntimeError(
                f"Dataset case {index} must include question and expected_answer."
            )

        cases.append(DatasetCase(question=question, expected_answer=expected_answer))

    return cases


def build_dataset_test_case(
    dataset_case: DatasetCase,
    actual_answer: str,
    context_chunks: list[str],
) -> LLMTestCase:
    """Build a DeepEval test case that includes the dataset reference answer."""
    return LLMTestCase(
        input=dataset_case.question,
        actual_output=actual_answer,
        expected_output=dataset_case.expected_answer,
        retrieval_context=context_chunks,
    )


def build_metrics() -> tuple[FaithfulnessMetric, GEval, GEval]:
    """Create RAG quality metrics plus expected-answer correctness."""
    return (
        FaithfulnessMetric(threshold=0.7),
        GEval(
            name="Answer Relevance",
            threshold=0.7,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
            ],
            criteria=(
                "Determine whether the actual output directly and completely "
                "answers the input question. Ignore minor wording differences. "
                "Penalize answers that are off-topic, evasive, or include "
                "irrelevant information."
            ),
        ),
        GEval(
            name="Correctness",
            threshold=0.7,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            criteria=(
                "Determine whether the actual output correctly answers the input "
                "and matches the meaning of the expected output. Do not require "
                "exact wording. Penalize missing, contradictory, or unsupported "
                "claims."
            ),
        ),
    )


def read_metric_value(metric: Any, attribute: str) -> Any:
    """Safely read metric fields that DeepEval fills after measurement."""
    return getattr(metric, attribute, None)


def retrieve_dataset_context(
    question: str,
    top_k: int,
    source: str | None = DEFAULT_SOURCE,
) -> list[str]:
    """Retrieve context, optionally constrained to one document source."""
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("No chunks found. Run `python src/ingest.py` first.")

    query_args: dict[str, Any] = {
        "query_texts": [question],
        "n_results": top_k,
    }
    if source:
        query_args["where"] = {"source": source}

    results = collection.query(**query_args)
    return results.get("documents", [[]])[0]


def answer_dataset_question(
    question: str,
    top_k: int,
    source: str | None = DEFAULT_SOURCE,
) -> tuple[str, list[str]]:
    """Answer a dataset question using the existing RAG generation function."""
    context_chunks = retrieve_dataset_context(
        question=question,
        top_k=top_k,
        source=source,
    )
    answer = generate_answer(question=question, context_chunks=context_chunks)
    return answer, context_chunks


def evaluate_case(
    case_number: int,
    dataset_case: DatasetCase,
    top_k: int,
    source: str | None,
) -> EvaluationResult:
    """Run RAG and DeepEval for a single dataset case."""
    actual_answer, context_chunks = answer_dataset_question(
        question=dataset_case.question,
        top_k=top_k,
        source=source,
    )
    test_case = build_dataset_test_case(
        dataset_case=dataset_case,
        actual_answer=actual_answer,
        context_chunks=context_chunks,
    )

    faithfulness_metric, answer_relevance_metric, correctness_metric = build_metrics()
    faithfulness_metric.measure(test_case)
    answer_relevance_metric.measure(test_case)
    correctness_metric.measure(test_case)

    return EvaluationResult(
        case_number=case_number,
        question=dataset_case.question,
        expected_answer=dataset_case.expected_answer,
        actual_answer=actual_answer,
        retrieval_context=context_chunks,
        faithfulness_score=read_metric_value(faithfulness_metric, "score"),
        faithfulness_success=read_metric_value(faithfulness_metric, "success"),
        faithfulness_reason=read_metric_value(faithfulness_metric, "reason") or "",
        answer_relevance_score=read_metric_value(answer_relevance_metric, "score"),
        answer_relevance_success=read_metric_value(answer_relevance_metric, "success"),
        answer_relevance_reason=read_metric_value(answer_relevance_metric, "reason") or "",
        correctness_score=read_metric_value(correctness_metric, "score"),
        correctness_success=read_metric_value(correctness_metric, "success"),
        correctness_reason=read_metric_value(correctness_metric, "reason") or "",
        evaluated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )


def run_dataset(
    dataset_path: Path = DATASET_PATH,
    top_k: int = 4,
    source: str | None = DEFAULT_SOURCE,
) -> list[EvaluationResult]:
    """Evaluate every dataset case with the existing RAG and DeepEval flow."""
    require_api_key()
    dataset_cases = load_dataset(dataset_path=dataset_path)
    results: list[EvaluationResult] = []

    for case_number, dataset_case in enumerate(dataset_cases, start=1):
        print(f"Evaluating case {case_number}/{len(dataset_cases)}: {dataset_case.question}")
        results.append(
            evaluate_case(
                case_number=case_number,
                dataset_case=dataset_case,
                top_k=top_k,
                source=source,
            )
        )

    return results


def save_report(results: list[EvaluationResult], report_path: Path = REPORT_PATH) -> Path:
    """Write evaluation results to reports/evaluation_report.csv."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_number",
        "question",
        "expected_answer",
        "actual_answer",
        "faithfulness_score",
        "faithfulness_success",
        "faithfulness_reason",
        "answer_relevance_score",
        "answer_relevance_success",
        "answer_relevance_reason",
        "correctness_score",
        "correctness_success",
        "correctness_reason",
        "retrieval_context",
        "evaluated_at",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case_number": result.case_number,
                    "question": result.question,
                    "expected_answer": result.expected_answer,
                    "actual_answer": result.actual_answer,
                    "faithfulness_score": result.faithfulness_score,
                    "faithfulness_success": result.faithfulness_success,
                    "faithfulness_reason": result.faithfulness_reason,
                    "answer_relevance_score": result.answer_relevance_score,
                    "answer_relevance_success": result.answer_relevance_success,
                    "answer_relevance_reason": result.answer_relevance_reason,
                    "correctness_score": result.correctness_score,
                    "correctness_success": result.correctness_success,
                    "correctness_reason": result.correctness_reason,
                    "retrieval_context": json.dumps(
                        result.retrieval_context,
                        ensure_ascii=False,
                    ),
                    "evaluated_at": result.evaluated_at,
                }
            )

    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full RAG evaluation dataset and write a CSV report."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Optional document source metadata filter. Use an empty value to search all documents.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="Path where the CSV evaluation report will be saved.",
    )
    args = parser.parse_args()

    source = args.source.strip() or None
    results = run_dataset(dataset_path=args.dataset, top_k=args.top_k, source=source)
    report_path = save_report(results=results, report_path=args.report)
    print(f"\nSaved evaluation report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
