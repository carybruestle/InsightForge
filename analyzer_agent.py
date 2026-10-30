from __future__ import annotations

import importlib
import json
import os
from typing import Any


class AnalyzerAgent:
    """Classifies what data resources are needed to answer a query."""

    SYSTEM_PROMPT = (
        "You are an analyst who determines what data resource is needed to answer the user's question: "
        "'sales_data', 'knowledge_base', 'both', or 'none'. "
        "sales_data contains actual sales data: Date,Product,Region,Sales,Customer_Age,Customer_Gender,Customer_Satisfaction. "
        "knowledge_base contains business knowledge and context but not raw data. "
        "You answer questions only pertaining to business trends, business models, sales analysis, and business insights.  "
        "If the question is out of bounds and not answerable by sales_data or knowledge_base, return 'none'. "
        "Return ONLY valid JSON in this format: {\"resource_requirement\": \"sales_data|knowledge_base|both|none\", "
        "\"reason\": \"short explanation\"}."
    )

    def _keyword_based_requirement(self, query: str) -> str | None:
        """Infer required resource from obvious query keywords."""
        normalized = query.lower()
        sales_data_terms = {
            "sales",
            "csv",
            "dataset",
            "data",
            "trend",
            "trends",
            "monthly",
            "quarterly",
            "yearly",
            "date",
            "region",
            "product",
            "customer",
            "satisfaction",
            "age",
            "gender",
        }
        knowledge_base_terms = {
            "knowledge base",
            "knowledge-base",
            "document",
            "documents",
            "pdf",
            "paper",
            "papers",
            "business model",
            "innovation",
            "insight",
            "insights",
            "recommendation",
            "recommendations",
        }

        sales_match = any(term in normalized for term in sales_data_terms)
        kb_match = any(term in normalized for term in knowledge_base_terms)

        if sales_match and kb_match:
            return "both"
        if sales_match:
            return "sales_data"
        if kb_match:
            return "knowledge_base"
        return None

    def analyze(self, query: str) -> dict[str, Any]:
        """Run AnalyzeAgent and return normalized classification metadata."""
        fallback: dict[str, Any] = {
            "resource_requirement": "both",
            "reason": "Fallback used because analyzer was unavailable.",
            "status": "fallback",
        }

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            fallback["status"] = "missing_api_key"
            return fallback

        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

        try:
            autogen_module = importlib.import_module("autogen")
            assistant_cls = getattr(autogen_module, "AssistantAgent")
            user_proxy_cls = getattr(autogen_module, "UserProxyAgent")

            config_item: dict[str, Any] = {"model": model, "api_key": api_key}
            if base_url:
                config_item["base_url"] = base_url

            analyzer = assistant_cls(
                name="AnalyzeAgent",
                llm_config={"config_list": [config_item], "temperature": 0},
                system_message=self.SYSTEM_PROMPT,
            )
            user_proxy = user_proxy_cls(
                name="analyze_user",
                human_input_mode="NEVER",
                code_execution_config=False,
                max_consecutive_auto_reply=0,
            )

            try:
                user_proxy.initiate_chat(analyzer, message=f"Question: {query}", silent=True)
            except TypeError:
                user_proxy.initiate_chat(analyzer, message=f"Question: {query}")
            last_message = analyzer.last_message() if hasattr(analyzer, "last_message") else None
            if not isinstance(last_message, dict):
                fallback["status"] = "invalid_analyzer_response"
                return fallback

            content = str(last_message.get("content", "")).strip()
            parsed = json.loads(content)
            resource_requirement = str(parsed.get("resource_requirement", "both")).strip().lower()
            reason = str(parsed.get("reason", "")).strip()

            if resource_requirement not in {"sales_data", "knowledge_base", "both", "none"}:
                resource_requirement = "both"

            keyword_requirement = self._keyword_based_requirement(query)
            if resource_requirement == "none" and keyword_requirement is not None:
                resource_requirement = keyword_requirement
                if reason:
                    reason = f"{reason} Adjusted by keyword guardrail."
                else:
                    reason = "Adjusted by keyword guardrail."

            return {
                "resource_requirement": resource_requirement,
                "reason": reason or "No reason provided.",
                "status": "ok",
                "raw_content": content,
            }
        except Exception as exc:
            fallback["status"] = "exception"
            fallback["error"] = str(exc)
            return fallback


def run_test_routine() -> None:
    """Run a quick manual test suite for AnalyzerAgent."""
    test_queries = [
        "Show monthly sales trends by region.",
        "What month shows the greatest sales?",
        "What is the average customer satisfaction for customers over 30 years old?",
        "What insights can you provide about our sales performance and how it relates to the business knowledge base?",
        "What business model innovation themes appear in the documents?",
        "Correlate customer satisfaction from sales_data with knowledge-base recommendations.",
        "Who won the 1998 FIFA World Cup?",
        "How do our sales trends compare to industry benchmarks mentioned in the knowledge base?",
        "How do our sales compare to Walmart's?"
    ]

    agent = AnalyzerAgent()
    print("AnalyzerAgent test routine")
    print("idx  requirement      status   query")
    print("---  ---------------  -------  ----------------------------------------------")
    for idx, query in enumerate(test_queries, start=1):
        result = agent.analyze(query)
        requirement = str(result.get("resource_requirement", "unknown"))
        status = str(result.get("status", "unknown"))
        reason = str(result.get("reason", ""))
        query_preview = query if len(query) <= 46 else f"{query[:43]}..."
        reason_preview = reason if len(reason) <= 80 else f"{reason[:77]}..."

        print(f"{idx:>3}  {requirement:<15}  {status:<7}  {query_preview}")
        print(f"     reason: {reason_preview}")


if __name__ == "__main__":
    run_test_routine()
