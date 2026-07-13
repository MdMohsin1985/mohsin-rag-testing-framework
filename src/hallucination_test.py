"""Run hallucination checks for questions that should not be answerable."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepeval.metrics import FaithfulnessMetric

from evaluate import build_test_case, require_api_key
from rag_app import answer_question


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "datasets" / "hallucination_tests.json"
REPORTS_DIR = ROOT_DIR / "reports"
REPORT_PATH = REPORTS_DIR / "hallucination_report.csv"
FAITHFULNESS_THRESHOLD = 0.7


@dataclass(frozen=True)
class HallucinationCase:
    """One unanswerable question used for hallucination testing."""

    question: str
    reason_unanswerable: str


@dataclass(frozen=True)
class HallucinationResult:
    """Serializable hallucination result for one test case."""

    case_number: int
    question: str
    reason_unanswerable: str
    actual_answer: str
    retrieval_context: list[str]
    faithfulness_score: float | None
    faithfulness_success: bool | None
    faithfulness_reason: str
    hallucinated: bool
    evaluated_at: str


def load_hallucination_cases(dataset_path: Path = DATASET_PATH) -> list[HallucinationCase]:
    """Read hallucination test cases from JSON."""
    if not dataset_path.exists():
        raise RuntimeError(f"Hallucination dataset not found: {dataset_path}")

    raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise RuntimeError("Hallucination dataset must be a JSON array.")

    cases: list[HallucinationCase] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise RuntimeError(f"Hallucination case {index} must be a JSON object.")

        question = str(raw_case.get("question", "")).strip()
        reason_unanswerable = str(raw_case.get("reason_unanswerable", "")).strip()
        if not question:
            raise RuntimeError(f"Hallucination case {index} must include question.")

        cases.append(
            HallucinationCase(
                question=question,
                reason_unanswerable=reason_unanswerable,
            )
        )

    return cases


def read_metric_value(metric: Any, attribute: str) -> Any:
    """Safely read metric fields that DeepEval fills after measurement."""
    return getattr(metric, attribute, None)


def evaluate_hallucination_case(
    case_number: int,
    test_case: HallucinationCase,
    top_k: int,
    threshold: float = FAITHFULNESS_THRESHOLD,
) -> HallucinationResult:
    """Run RAG and flag hallucination when the answer is not faithful."""
    answer, context_chunks = answer_question(question=test_case.question, top_k=top_k)
    deepeval_case = build_test_case(
        question=test_case.question,
        answer=answer,
        context_chunks=context_chunks,
    )

    faithfulness_metric = FaithfulnessMetric(threshold=threshold)
    faithfulness_metric.measure(deepeval_case)

    faithfulness_score = read_metric_value(faithfulness_metric, "score")
    faithfulness_success = read_metric_value(faithfulness_metric, "success")
    hallucinated = faithfulness_success is False

    return HallucinationResult(
        case_number=case_number,
        question=test_case.question,
        reason_unanswerable=test_case.reason_unanswerable,
        actual_answer=answer,
        retrieval_context=context_chunks,
        faithfulness_score=faithfulness_score,
        faithfulness_success=faithfulness_success,
        faithfulness_reason=read_metric_value(faithfulness_metric, "reason") or "",
        hallucinated=hallucinated,
        evaluated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )


def run_hallucination_tests(
    dataset_path: Path = DATASET_PATH,
    top_k: int = 4,
    threshold: float = FAITHFULNESS_THRESHOLD,
) -> list[HallucinationResult]:
    """Evaluate all hallucination test cases with the existing RAG flow."""
    require_api_key()
    test_cases = load_hallucination_cases(dataset_path=dataset_path)
    results: list[HallucinationResult] = []

    for case_number, test_case in enumerate(test_cases, start=1):
        print(f"Evaluating hallucination case {case_number}/{len(test_cases)}: {test_case.question}")
        results.append(
            evaluate_hallucination_case(
                case_number=case_number,
                test_case=test_case,
                top_k=top_k,
                threshold=threshold,
            )
        )

    return results


def save_report(
    results: list[HallucinationResult],
    report_path: Path = REPORT_PATH,
) -> Path:
    """Write hallucination results to reports/hallucination_report.csv."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_number",
        "question",
        "reason_unanswerable",
        "actual_answer",
        "faithfulness_score",
        "faithfulness_success",
        "faithfulness_reason",
        "hallucinated",
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
                    "reason_unanswerable": result.reason_unanswerable,
                    "actual_answer": result.actual_answer,
                    "faithfulness_score": result.faithfulness_score,
                    "faithfulness_success": result.faithfulness_success,
                    "faithfulness_reason": result.faithfulness_reason,
                    "hallucinated": result.hallucinated,
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
        description="Run hallucination tests and write a CSV report."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the hallucination dataset JSON file.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=FAITHFULNESS_THRESHOLD,
        help="Faithfulness threshold below which an answer is flagged as hallucinated.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="Path where the hallucination CSV report will be saved.",
    )
    args = parser.parse_args()

    results = run_hallucination_tests(
        dataset_path=args.dataset,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    report_path = save_report(results=results, report_path=args.report)
    hallucinated_count = sum(1 for result in results if result.hallucinated)
    print(f"\nHallucinated responses: {hallucinated_count}/{len(results)}")
    print(f"Saved hallucination report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
