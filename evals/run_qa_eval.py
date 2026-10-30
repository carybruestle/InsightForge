from __future__ import annotations

import argparse
import json
import sys
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents_orchestrator import InsightForgeOrchestrator
from agents_orchestrator import _load_all_vector_stores


VALID_ROUTES = {"sales_data", "knowledge_base", "both", "none"}


@dataclass
class EvalRow:
    """Single evaluation row loaded from JSONL."""

    id: str
    question: str
    reference_answer: str
    expected_route: str | None
    tags: list[str]
    difficulty: str | None


class TokenEstimator:
    """Estimate token usage with tiktoken when available."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._encoder = None

        try:
            import tiktoken

            try:
                self._encoder = tiktoken.encoding_for_model(model_name)
            except Exception:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is None:
            return max(1, int(len(text) / 4))
        return len(self._encoder.encode(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QAEvalChain against InsightForge outputs.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/datasets/qa_gold.jsonl"),
        help="Path to JSONL dataset.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("indexes"),
        help="Directory containing FAISS indexes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evals/reports"),
        help="Output directory for report files.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional cap on number of dataset rows to evaluate.",
    )
    parser.add_argument(
        "--price-input-per-1k",
        type=float,
        default=0.0,
        help="Optional estimated input token price in USD per 1k tokens.",
    )
    parser.add_argument(
        "--price-output-per-1k",
        type=float,
        default=0.0,
        help="Optional estimated output token price in USD per 1k tokens.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> list[EvalRow]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    rows: list[EvalRow] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            data = json.loads(line)
            question = str(data.get("question", "")).strip()
            if not question:
                raise ValueError(f"Missing question at line {line_number}")

            reference_answer = str(data.get("reference_answer", "")).strip()
            expected_route_raw = str(data.get("expected_route", "")).strip().lower()
            expected_route = expected_route_raw or None
            if expected_route is not None and expected_route not in VALID_ROUTES:
                raise ValueError(
                    f"Invalid expected_route '{expected_route}' at line {line_number}; "
                    f"valid values are {sorted(VALID_ROUTES)}"
                )

            tags_raw = data.get("tags", [])
            tags = [str(tag) for tag in tags_raw] if isinstance(tags_raw, list) else []
            difficulty_raw = str(data.get("difficulty", "")).strip()
            difficulty = difficulty_raw or None

            rows.append(
                EvalRow(
                    id=str(data.get("id", f"row_{line_number}")),
                    question=question,
                    reference_answer=reference_answer,
                    expected_route=expected_route,
                    tags=tags,
                    difficulty=difficulty,
                )
            )

    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset_path}")

    return rows


def load_qa_evaluator() -> tuple[Any | None, str]:
    """Load a QA evaluator with broad compatibility across LangChain versions."""
    try:
        from langchain.evaluation import load_evaluator

        evaluator = load_evaluator("qa")
        return evaluator, "langchain.evaluation.load_evaluator('qa')"
    except Exception:
        pass

    try:
        from langchain.evaluation.qa import QAEvalChain
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(temperature=0)
        evaluator = QAEvalChain.from_llm(llm=llm)
        return evaluator, "langchain.evaluation.qa.QAEvalChain"
    except Exception:
        return None, "unavailable"


def evaluate_with_chain(
    evaluator: Any,
    question: str,
    prediction: str,
    reference_answer: str,
) -> tuple[str | None, float | None, str]:
    """Return normalized verdict, numeric score when possible, and rationale text."""
    try:
        result = evaluator.evaluate_strings(
            input=question,
            prediction=prediction,
            reference=reference_answer,
        )
    except Exception as exc:
        return None, None, f"evaluation_error: {exc}"

    if not isinstance(result, dict):
        return None, None, f"unexpected_evaluator_output: {result}"

    reasoning = str(
        result.get("reasoning")
        or result.get("text")
        or result.get("comment")
        or ""
    ).strip()

    score_value = result.get("score")
    if isinstance(score_value, (int, float)):
        numeric = float(score_value)
        if numeric >= 0.5:
            return "CORRECT", numeric, reasoning
        return "INCORRECT", numeric, reasoning

    verdict_text = str(
        result.get("value")
        or result.get("grade")
        or result.get("verdict")
        or score_value
        or ""
    ).strip().upper()

    if "CORRECT" in verdict_text or verdict_text in {"Y", "YES", "PASS", "TRUE"}:
        return "CORRECT", 1.0, reasoning
    if "INCORRECT" in verdict_text or verdict_text in {"N", "NO", "FAIL", "FALSE"}:
        return "INCORRECT", 0.0, reasoning

    return None, None, reasoning


def compute_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _format_bucket_table(title: str, bucket_stats: dict[str, dict[str, int]]) -> list[str]:
    lines = [f"## {title}"]
    if not bucket_stats:
        lines.append("- No rows available.")
        lines.append("")
        return lines

    lines.extend(
        [
            "| Bucket | Total | Judged | Correct | Pass Rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bucket in sorted(bucket_stats.keys()):
        stats = bucket_stats[bucket]
        total = stats.get("total", 0)
        judged = stats.get("judged", 0)
        correct = stats.get("correct", 0)
        pass_rate = (correct / judged) if judged else 0.0
        lines.append(f"| {bucket} | {total} | {judged} | {correct} | {pass_rate:.2%} |")
    lines.append("")
    return lines


def build_report_markdown(
    *,
    run_timestamp: str,
    dataset_path: Path,
    results_path: Path,
    evaluator_name: str,
    total_rows: int,
    judged_rows: int,
    correct_rows: int,
    route_labeled_rows: int,
    route_match_rows: int,
    latency_ms_values: list[float],
    input_tokens: list[int],
    output_tokens: list[int],
    estimated_costs: list[float],
    per_route_stats: dict[str, dict[str, int]],
    per_difficulty_stats: dict[str, dict[str, int]],
) -> str:
    avg_latency = statistics.mean(latency_ms_values) if latency_ms_values else 0.0
    p50_latency = compute_percentile(latency_ms_values, 0.5)
    p95_latency = compute_percentile(latency_ms_values, 0.95)
    pass_rate = (correct_rows / judged_rows) if judged_rows else 0.0
    route_accuracy = (route_match_rows / route_labeled_rows) if route_labeled_rows else 0.0

    total_input_tokens = sum(input_tokens)
    total_output_tokens = sum(output_tokens)
    avg_input_tokens = statistics.mean(input_tokens) if input_tokens else 0.0
    avg_output_tokens = statistics.mean(output_tokens) if output_tokens else 0.0
    total_estimated_cost = sum(estimated_costs)

    lines = [
        "# InsightForge QA Evaluation Report",
        "",
        f"- Run timestamp (UTC): {run_timestamp}",
        f"- Dataset: {dataset_path}",
        f"- Results file: {results_path}",
        f"- Evaluator: {evaluator_name}",
        "",
        "## Efficacy",
        f"- Total rows: {total_rows}",
        f"- Judged rows (with reference answers): {judged_rows}",
        f"- Correct rows: {correct_rows}",
        f"- Pass rate: {pass_rate:.2%}",
        "",
        "## Efficiency",
        f"- Avg latency: {avg_latency:.1f} ms",
        f"- p50 latency: {p50_latency:.1f} ms",
        f"- p95 latency: {p95_latency:.1f} ms",
        f"- Avg estimated input tokens/query: {avg_input_tokens:.1f}",
        f"- Avg estimated output tokens/query: {avg_output_tokens:.1f}",
        f"- Total estimated input tokens: {total_input_tokens}",
        f"- Total estimated output tokens: {total_output_tokens}",
        f"- Total estimated cost (USD): ${total_estimated_cost:.6f}",
        "",
        "## Route Quality",
        f"- Rows with expected_route label: {route_labeled_rows}",
        f"- Matching routes: {route_match_rows}",
        f"- Route accuracy: {route_accuracy:.2%}",
        "",
        "## Notes",
        "- Token counts are estimates from visible text (question and model response) and may undercount full prompt/context tokens used internally by agents.",
        "- If evaluator is unavailable, judged_rows will be 0 and this run reports efficiency and route metrics only.",
    ]

    lines.extend(["", *_format_bucket_table("By Route", per_route_stats)])
    lines.extend(["", *_format_bucket_table("By Difficulty", per_difficulty_stats)])

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    dataset_rows = load_dataset(args.dataset)
    if args.sample_size > 0:
        dataset_rows = dataset_rows[: args.sample_size]

    project_root = Path(__file__).resolve().parent.parent
    index_dir = (project_root / args.index_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_stores = _load_all_vector_stores(index_dir)
    source_names = sorted(vector_stores.keys())

    def route_all_sources(query: str, names: list[str]) -> tuple[list[str], dict[str, Any]]:
        return (
            names,
            {
                "query": query,
                "available_sources": names,
                "selected_sources": names,
                "raw_router_content": "",
                "router_status": "eval_route_all_sources",
            },
        )

    orchestrator = InsightForgeOrchestrator(
        route_sources=route_all_sources,
        search_sources=lambda _query, _stores: [],
        build_reply=lambda _query, _results, _sources: "",
    )

    evaluator, evaluator_name = load_qa_evaluator()

    model_name = "gpt-4o-mini"
    try:
        model_name = str(orchestrator._model_name)
    except Exception:
        pass

    token_estimator = TokenEstimator(model_name=model_name)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    detailed_results_path = output_dir / f"qa_eval_results_{run_timestamp}.jsonl"
    summary_report_path = output_dir / f"qa_eval_report_{run_timestamp}.md"

    judged_rows = 0
    correct_rows = 0
    route_labeled_rows = 0
    route_match_rows = 0

    per_route_stats: dict[str, dict[str, int]] = {}
    per_difficulty_stats: dict[str, dict[str, int]] = {}

    latency_ms_values: list[float] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    estimated_costs: list[float] = []

    def _ensure_bucket(bucket: str, stats: dict[str, dict[str, int]]) -> dict[str, int]:
        if bucket not in stats:
            stats[bucket] = {"total": 0, "judged": 0, "correct": 0}
        return stats[bucket]

    with detailed_results_path.open("w", encoding="utf-8") as output_handle:
        for row in dataset_rows:
            start = time.perf_counter()
            result = orchestrator.execute_query(
                query=row.question,
                source_names=source_names,
                vector_stores=vector_stores,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            prediction = result.reply
            actual_route = str(result.debug.get("resource_requirement", "")).strip().lower() or None

            if row.expected_route:
                route_labeled_rows += 1
                if actual_route == row.expected_route:
                    route_match_rows += 1

            route_bucket_name = row.expected_route or "unlabeled"
            difficulty_bucket_name = row.difficulty or "unspecified"
            route_bucket = _ensure_bucket(route_bucket_name, per_route_stats)
            difficulty_bucket = _ensure_bucket(difficulty_bucket_name, per_difficulty_stats)
            route_bucket["total"] += 1
            difficulty_bucket["total"] += 1

            input_token_estimate = token_estimator.estimate(row.question)
            output_token_estimate = token_estimator.estimate(prediction)
            estimated_cost = (
                (input_token_estimate / 1000.0) * args.price_input_per_1k
                + (output_token_estimate / 1000.0) * args.price_output_per_1k
            )

            latency_ms_values.append(elapsed_ms)
            input_tokens.append(input_token_estimate)
            output_tokens.append(output_token_estimate)
            estimated_costs.append(estimated_cost)

            verdict: str | None = None
            numeric_score: float | None = None
            reasoning = ""

            if evaluator is not None and row.reference_answer:
                verdict, numeric_score, reasoning = evaluate_with_chain(
                    evaluator=evaluator,
                    question=row.question,
                    prediction=prediction,
                    reference_answer=row.reference_answer,
                )
                if verdict is not None:
                    judged_rows += 1
                    route_bucket["judged"] += 1
                    difficulty_bucket["judged"] += 1
                    if verdict == "CORRECT":
                        correct_rows += 1
                        route_bucket["correct"] += 1
                        difficulty_bucket["correct"] += 1

            out_row = {
                "id": row.id,
                "question": row.question,
                "reference_answer": row.reference_answer,
                "prediction": prediction,
                "expected_route": row.expected_route,
                "actual_route": actual_route,
                "route_match": actual_route == row.expected_route if row.expected_route else None,
                "latency_ms": elapsed_ms,
                "estimated_input_tokens": input_token_estimate,
                "estimated_output_tokens": output_token_estimate,
                "estimated_cost_usd": estimated_cost,
                "qa_verdict": verdict,
                "qa_score": numeric_score,
                "qa_reasoning": reasoning,
                "debug": result.debug,
            }
            output_handle.write(json.dumps(out_row, ensure_ascii=True) + "\n")

    summary_text = build_report_markdown(
        run_timestamp=run_timestamp,
        dataset_path=args.dataset,
        results_path=detailed_results_path,
        evaluator_name=evaluator_name,
        total_rows=len(dataset_rows),
        judged_rows=judged_rows,
        correct_rows=correct_rows,
        route_labeled_rows=route_labeled_rows,
        route_match_rows=route_match_rows,
        latency_ms_values=latency_ms_values,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_costs=estimated_costs,
        per_route_stats=per_route_stats,
        per_difficulty_stats=per_difficulty_stats,
    )
    summary_report_path.write_text(summary_text, encoding="utf-8")

    print("InsightForge QA eval completed.")
    print(f"Evaluator: {evaluator_name}")
    print(f"Rows evaluated: {len(dataset_rows)}")
    print(f"Detailed results: {detailed_results_path}")
    print(f"Summary report: {summary_report_path}")


if __name__ == "__main__":
    main()
