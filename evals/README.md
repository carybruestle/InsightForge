# Evaluation Workflow

This folder adds offline QA evaluation for InsightForge using QAEvalChain-compatible evaluators.

## Files

- `datasets/qa_gold.jsonl`: Gold evaluation set (JSONL).
- `run_qa_eval.py`: Runner that executes orchestrator queries and writes efficacy/efficiency reports.
- `reports/`: Generated run outputs.

## Dataset Format

Each JSONL row supports:

- `id`: unique identifier.
- `question`: user-style query.
- `reference_answer`: expected answer text for QA grading.
- `expected_route`: one of `sales_data`, `knowledge_base`, `both`, `none`.
- `difficulty`: optional label.
- `tags`: optional string list.

If `reference_answer` is blank, the row is still measured for efficiency and route accuracy, but not QA correctness.

## Run

From project root:

```powershell
python evals/run_qa_eval.py --dataset evals/datasets/qa_gold.jsonl
```

Optional cost estimation inputs:

```powershell
python evals/run_qa_eval.py \
  --dataset evals/datasets/qa_gold.jsonl \
  --price-input-per-1k 0.00015 \
  --price-output-per-1k 0.00060
```

Optional subset run:

```powershell
python evals/run_qa_eval.py --sample-size 10
```

## Outputs

- `evals/reports/qa_eval_results_<timestamp>.jsonl`: Per-row details including answer, verdict, latency, token estimates, route match, and debug metadata.
- `evals/reports/qa_eval_report_<timestamp>.md`: Summary report with pass rate, latency distribution, token/cost estimates, and route accuracy.

The summary report also includes breakdown tables:

- by expected route (`sales_data`, `knowledge_base`, `both`, `none`)
- by difficulty (`easy`, `medium`, `hard`)

## Notes

- The evaluator loader first tries `langchain.evaluation.load_evaluator("qa")`.
- If unavailable, it falls back to `QAEvalChain` construction when possible.
- Token counts are estimates from visible question/response text and may undercount full internal prompt/context token usage.
- The starter `qa_gold.jsonl` now includes a mix of grounded numeric references for sales-data questions and behavior-focused references for KB/both questions. Tighten these references over time as you curate stricter expected outputs.
