from __future__ import annotations

import csv
import importlib
import json
import os
from pathlib import Path
from typing import Any

from sales_data_context import serialize_sales_context, build_sales_context


class SalesAgent:
    """Answers sales-data questions using an AutoGen assistant."""

    SYSTEM_PROMPT = (
        "You are an interpreter of sales data. You answer user questions using only the provided sales_data context. "
        "sales_data contains actual sales data: Date,Product,Region,Sales,Customer_Age,Customer_Gender,Customer_Satisfaction. "
        "Do not invent fields or facts not present in the context. "
        "If the data is insufficient, clearly state what is missing. "
        "Provide concise, business-focused answers."
    )

    def respond(self, query: str, sales_data: Any) -> str:
        """Return an answer for query using the provided sales_data context."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "SalesAgent is unavailable because OPENAI_API_KEY is not set."

        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

        context = build_sales_context(query=query, sales_data=sales_data, max_sample_rows=240)
        total_rows = context.get("row_count_total", 0)
        filtered_rows = context.get("row_count_filtered", 0)
        serialized_data = json.dumps(context, ensure_ascii=True)
        print(f"[SalesAgent] Query: {query[:60]} | Rows: {filtered_rows}/{total_rows} | Context: {len(serialized_data)} bytes", flush=True)

        try:
            autogen_module = importlib.import_module("autogen")
            assistant_cls = getattr(autogen_module, "AssistantAgent")
            user_proxy_cls = getattr(autogen_module, "UserProxyAgent")

            config_item: dict[str, Any] = {"model": model, "api_key": api_key}
            if base_url:
                config_item["base_url"] = base_url

            assistant = assistant_cls(
                name="SalesAgent",
                llm_config={"config_list": [config_item], "temperature": 0},
                system_message=self.SYSTEM_PROMPT,
            )
            user_proxy = user_proxy_cls(
                name="sales_user",
                human_input_mode="NEVER",
                code_execution_config=False,
                max_consecutive_auto_reply=0,
            )

            prompt = (
                f"User query: {query}\n\n"
                "sales_data context (JSON with deterministic filters, aggregates, and sampled rows):\n"
                f"{serialized_data}\n\n"
                "Answer the query using this data only."
            )

            try:
                user_proxy.initiate_chat(assistant, message=prompt, silent=True)
            except TypeError:
                user_proxy.initiate_chat(assistant, message=prompt)

            last_message = assistant.last_message() if hasattr(assistant, "last_message") else None
            if isinstance(last_message, dict):
                content = str(last_message.get("content", "")).strip()
                if content:
                    return content

            return "SalesAgent could not generate a response from the provided sales_data."
        except Exception as exc:
            return f"SalesAgent failed to run: {exc}"


def load_sales_data_from_csv(csv_path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    """Load sales_data.csv into a list of row dictionaries."""
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            rows.append(dict(row))
            if max_rows is not None and idx >= max_rows:
                break
    return rows


def run_test_routine() -> None:
    """Run a quick manual test suite for SalesAgent."""
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "Data" / "sales_data.csv"
    if not csv_path.exists():
        print(f"sales_data.csv not found: {csv_path}")
        return

    sales_data = load_sales_data_from_csv(csv_path)

    test_queries = [
        "Show monthly sales trends by region.",
        "What month shows the greatest sales?",
        "What is the average customer satisfaction for customers over 30 years old?",
        "Which product sold the most with customers under 25 years old?",
        "Compare sales performance between regions.",
        ]

    agent = SalesAgent()
    print("SalesAgent test routine")
    print(f"Using data file: {csv_path}")
    print(f"Rows used for context: {len(sales_data)}")
    print("-" * 80)

    for idx, query in enumerate(test_queries, start=1):
        response = agent.respond(query, sales_data)
        query_preview = query if len(query) <= 46 else f"{query[:43]}..."
        response_preview = response if len(response) <= 92 else f"{response[:89]}..."
        print(f"{idx:>3}. {query_preview}")
        print(f"    Response: {response_preview}\n")


if __name__ == "__main__":
    run_test_routine()
