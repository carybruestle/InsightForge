from __future__ import annotations

import csv
import importlib
import json
import os
from pathlib import Path
from typing import Any

from sales_data_context import serialize_sales_context


class BothAgent:
    """Answers cross-resource questions using sales_data plus KB chunks."""

    SYSTEM_PROMPT = (
        "You are a business analysis assistant. Use both provided resources: "
        "(1) sales_data context and (2) knowledge-base chunks. "
        "Do not invent facts outside the provided context. "
        "If one resource is insufficient, say what is missing. "
        "Provide concise, business-focused answers and do not mention source names or document titles."
    )

    def _retrieve_top_chunks(
        self,
        query: str,
        vector_stores: dict[str, Any],
        k: int = 3,
    ) -> list[dict[str, str]]:
        """Retrieve top-k chunks globally ranked across stores by similarity score."""
        scored_chunks: list[tuple[float, dict[str, str]]] = []
        for source_name, store in vector_stores.items():
            try:
                docs_with_scores = store.similarity_search_with_score(query, k=k)
                for doc, score in docs_with_scores:
                    content = (getattr(doc, "page_content", "") or "").strip()
                    if not content:
                        continue
                    scored_chunks.append(
                        (
                            float(score),
                            {"source": source_name, "content": content},
                        )
                    )
            except Exception:
                try:
                    docs = store.similarity_search(query, k=k)
                except Exception:
                    continue
                for doc in docs:
                    content = (getattr(doc, "page_content", "") or "").strip()
                    if not content:
                        continue
                    scored_chunks.append(
                        (
                            float("inf"),
                            {"source": source_name, "content": content},
                        )
                    )

        scored_chunks.sort(key=lambda item: item[0])
        return [chunk for _, chunk in scored_chunks[:k]]

    def respond(
        self,
        query: str,
        sales_data: Any,
        vector_stores: dict[str, Any],
        top_k_chunks: int = 3,
    ) -> str:
        """Answer query using sales_data and top-k KB chunks."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "BothAgent is unavailable because OPENAI_API_KEY is not set."

        if not vector_stores:
            return "BothAgent found no available knowledge-base indexes."

        kb_chunks = self._retrieve_top_chunks(query, vector_stores, k=top_k_chunks)
        if not kb_chunks:
            return "BothAgent found no relevant knowledge-base chunks for this query."

        source_names = sorted({chunk["source"] for chunk in kb_chunks})
        print(f"[BothAgent] Retrieved sources: {', '.join(source_names)}")

        serialized_data = serialize_sales_context(query=query, sales_data=sales_data, max_sample_rows=240)
        kb_chunks_text = "\n".join(chunk["content"] for chunk in kb_chunks)

        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

        try:
            autogen_module = importlib.import_module("autogen")
            assistant_cls = getattr(autogen_module, "AssistantAgent")
            user_proxy_cls = getattr(autogen_module, "UserProxyAgent")

            config_item: dict[str, Any] = {"model": model, "api_key": api_key}
            if base_url:
                config_item["base_url"] = base_url

            assistant = assistant_cls(
                name="BothAgent",
                llm_config={"config_list": [config_item], "temperature": 0},
                system_message=self.SYSTEM_PROMPT,
            )
            user_proxy = user_proxy_cls(
                name="both_user",
                human_input_mode="NEVER",
                code_execution_config=False,
                max_consecutive_auto_reply=0,
            )

            prompt = (
                f"User query: {query}\n\n"
                "sales_data context (JSON with deterministic filters, aggregates, and sampled rows):\n"
                f"{serialized_data}\n\n"
                f"Knowledge-base chunks (top {top_k_chunks} by similarity):\n"
                f"{kb_chunks_text}\n\n"
                "Answer using both resources when relevant."
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

            return "BothAgent could not generate a response from the provided contexts."
        except Exception as exc:
            return f"BothAgent failed to run: {exc}"


def load_sales_data_from_csv(csv_path: Path, max_rows: int = 200) -> list[dict[str, str]]:
    """Load sales_data.csv into a list of row dictionaries."""
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            rows.append(dict(row))
            if idx >= max_rows:
                break
    return rows


def run_test_routine() -> None:
    """Run a quick manual test suite for BothAgent."""
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "Data" / "sales_data.csv"
    index_dir = project_root / "indexes"

    if not csv_path.exists():
        print(f"sales_data.csv not found: {csv_path}")
        return

    if not index_dir.exists():
        print(f"Index directory not found: {index_dir}")
        return

    sales_data = load_sales_data_from_csv(csv_path)

    vector_stores: dict[str, Any] = {}
    try:
        embeddings_module = importlib.import_module("embeddings")
        load_vector_store = getattr(embeddings_module, "load_vector_store")
        for path in sorted(index_dir.iterdir()):
            if not path.is_dir():
                continue
            try:
                vector_stores[path.name] = load_vector_store(index_dir, path.name)
            except Exception:
                continue
    except Exception as exc:
        print(f"Failed to load vector stores: {exc}")
        return

    if not vector_stores:
        print(f"No vector stores found under: {index_dir}")
        return

    test_queries = [
        "Compare our monthly regional sales trends with strategic recommendations from the knowledge base.",
        "Which low-satisfaction product segments should we prioritize, based on what the documents suggest?",
        "Using sales patterns and the indexed business guidance, what actions could improve next-quarter performance?",
        "How do age-group sales differences align with customer-focused recommendations in the knowledge base?",
        "Correlate customer satisfaction trends in sales_data with themes from the business model innovation documents.",
    ]

    agent = BothAgent()
    print("BothAgent test routine")
    print(f"Using data file: {csv_path}")
    print(f"Rows used for context: {len(sales_data)}")
    print(f"Vector stores loaded: {len(vector_stores)}")
    print("Top KB chunks per query: 3")
    print("-" * 80)

    for idx, query in enumerate(test_queries, start=1):
        response = agent.respond(query, sales_data, vector_stores, top_k_chunks=3)
        print(f"{idx:>3}. {query}")
        print(f"    Response: {response}\n")


if __name__ == "__main__":
    run_test_routine()
