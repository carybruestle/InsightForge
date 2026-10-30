from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from analyzer_agent import AnalyzerAgent
from both_agent import BothAgent
from embeddings import load_vector_store
from knowledge_base_agent import KnowledgeBaseAgent
from sales_agent import SalesAgent
from sales_agent import load_sales_data_from_csv


State = dict[str, Any]
RouteFn = Callable[[str, list[str]], tuple[list[str], dict[str, Any]]]
SearchFn = Callable[[str, dict[str, Any]], list[dict[str, str]]]
ReplyFn = Callable[[str, list[dict[str, str]], list[str]], str]


@dataclass
class OrchestratorResult:
    """Container for orchestrator output and traceable metadata."""

    reply: str
    debug: dict[str, Any]


class InsightForgeOrchestrator:
    """Coordinates analyzer-driven dispatch across specialized agents.

    The orchestrator prefers running through a LangGraph workflow when available,
    and falls back to a deterministic linear flow otherwise.
    """

    OUT_OF_SCOPE_REPLY = (
        "I'm sorry, but I can't answer your question with the available resources. "
        "I can only help with business trends and insights that are answerable from the sales data "
        "and the indexed business knowledge base."
    )

    def __init__(
        self,
        route_sources: RouteFn,
        search_sources: SearchFn,
        build_reply: ReplyFn,
    ) -> None:
        self._route_sources = route_sources
        self._search_sources = search_sources
        self._build_reply = build_reply
        self._analyzer_agent = AnalyzerAgent()
        self._sales_agent = SalesAgent()
        self._knowledge_base_agent = KnowledgeBaseAgent()
        self._both_agent = BothAgent()
        self._sales_data_path = Path(__file__).resolve().parent / "Data" / "sales_data.csv"

        self._model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self._langgraph_available = False
        self._compiled_graph: Any | None = None
        self._graph_error: str | None = None

        self._try_build_graph()

    def _try_build_graph(self) -> None:
        """Build the graph if LangGraph is installed and importable."""
        try:
            graph_module = importlib.import_module("langgraph.graph")
            state_graph_cls = getattr(graph_module, "StateGraph")
            end = getattr(graph_module, "END")

            graph = state_graph_cls(dict)
            graph.add_node("analyze", self._analyze_node)
            graph.add_node("sales_agent", self._sales_node)
            graph.add_node("knowledge_base_agent", self._knowledge_base_node)
            graph.add_node("both_agent", self._both_node)
            graph.add_node("out_of_scope", self._out_of_scope_node)
            graph.set_entry_point("analyze")
            graph.add_conditional_edges(
                "analyze",
                self._next_node_from_requirement,
                {
                    "sales_agent": "sales_agent",
                    "knowledge_base_agent": "knowledge_base_agent",
                    "both_agent": "both_agent",
                    "out_of_scope": "out_of_scope",
                },
            )
            graph.add_edge("sales_agent", end)
            graph.add_edge("knowledge_base_agent", end)
            graph.add_edge("both_agent", end)
            graph.add_edge("out_of_scope", end)

            self._compiled_graph = graph.compile()
            self._langgraph_available = True
        except Exception as exc:
            self._langgraph_available = False
            self._compiled_graph = None
            self._graph_error = str(exc)

    def _analyze_node(self, state: State) -> State:
        query = str(state.get("query", ""))
        analyze_debug = self._analyzer_agent.analyze(query)
        return {
            **state,
            "resource_requirement": analyze_debug.get("resource_requirement", "both"),
            "analyze_debug": analyze_debug,
        }

    def _next_node_from_requirement(self, state: State) -> str:
        resource_requirement = str(state.get("resource_requirement", "both")).strip().lower()
        if resource_requirement == "none":
            return "out_of_scope"
        if resource_requirement == "sales_data":
            return "sales_agent"
        if resource_requirement == "knowledge_base":
            return "knowledge_base_agent"
        return "both_agent"

    def _out_of_scope_node(self, state: State) -> State:
        return {
            **state,
            "agent_used": "out_of_scope",
            "selected_sources": [],
            "router_debug": {
                "router_status": "skipped_out_of_scope",
                "selected_sources": [],
            },
            "reply": self.OUT_OF_SCOPE_REPLY,
        }

    def _load_sales_data(self) -> list[dict[str, str]]:
        try:
            return load_sales_data_from_csv(self._sales_data_path)
        except Exception:
            return []

    def _sales_node(self, state: State) -> State:
        query = str(state.get("query", ""))
        sales_data = self._load_sales_data()
        reply = self._sales_agent.respond(query, sales_data)
        return {
            **state,
            "agent_used": "sales_agent",
            "selected_sources": [],
            "router_debug": {
                "router_status": "not_applicable_sales_only",
                "selected_sources": [],
            },
            "reply": reply,
        }

    def _knowledge_base_node(self, state: State) -> State:
        query = str(state.get("query", ""))
        reply = self._knowledge_base_agent.respond(query)
        return {
            **state,
            "agent_used": "knowledge_base_agent",
            "selected_sources": [],
            "router_debug": {
                "router_status": "not_applicable_kb_only",
                "selected_sources": [],
            },
            "reply": reply,
        }

    def _both_node(self, state: State) -> State:
        query = str(state.get("query", ""))
        source_names = list(state.get("source_names", []))
        vector_stores = dict(state.get("vector_stores", {}))

        try:
            selected_sources, router_debug = self._route_sources(query, source_names)
        except Exception as exc:
            selected_sources = source_names
            router_debug = {
                "query": query,
                "available_sources": source_names,
                "selected_sources": selected_sources,
                "raw_router_content": "",
                "router_status": "exception",
                "error": str(exc),
            }

        if selected_sources:
            active_stores = {
                name: vector_stores[name]
                for name in selected_sources
                if name in vector_stores
            }
        else:
            active_stores = vector_stores

        sales_data = self._load_sales_data()

        reply = self._both_agent.respond(
            query=query,
            sales_data=sales_data,
            vector_stores=active_stores,
            top_k_chunks=3,
        )

        return {
            **state,
            "agent_used": "both_agent",
            "selected_sources": selected_sources,
            "router_debug": router_debug,
            "reply": reply,
        }

    def _run_linear(self, initial_state: State) -> State:
        analyzed_state = self._analyze_node(initial_state)
        next_node = self._next_node_from_requirement(analyzed_state)
        if next_node == "sales_agent":
            return self._sales_node(analyzed_state)
        if next_node == "knowledge_base_agent":
            return self._knowledge_base_node(analyzed_state)
        if next_node == "both_agent":
            return self._both_node(analyzed_state)
        return self._out_of_scope_node(analyzed_state)

    def execute_query(
        self,
        query: str,
        source_names: list[str],
        vector_stores: dict[str, Any],
    ) -> OrchestratorResult:
        """Execute the orchestration workflow and return reply plus debug metadata."""
        initial_state: State = {
            "query": query,
            "source_names": source_names,
            "vector_stores": vector_stores,
        }

        debug: dict[str, Any] = {
            "orchestrator": "InsightForgeOrchestrator",
            "model": self._model_name,
            "langgraph_available": self._langgraph_available,
            "execution_mode": "langgraph" if self._langgraph_available else "linear",
            "graph_error": self._graph_error,
            "status": "ok",
        }

        try:
            if self._compiled_graph is not None:
                final_state = self._compiled_graph.invoke(initial_state)
            else:
                final_state = self._run_linear(initial_state)

            analyze_debug = dict(final_state.get("analyze_debug", {}))
            router_debug = dict(final_state.get("router_debug", {}))
            debug["resource_requirement"] = str(final_state.get("resource_requirement", "both"))
            debug["agent_used"] = str(final_state.get("agent_used", "unknown"))
            debug["analyze_debug"] = analyze_debug
            debug["router_debug"] = router_debug
            debug["selected_sources"] = list(final_state.get("selected_sources", []))
            debug["result_count"] = int(len(debug["selected_sources"]))

            reply = str(final_state.get("reply", "")).strip()
            if not reply:
                debug["status"] = "empty_reply"
                reply = (
                    "I could not generate a response from the current context. "
                    "Try refining your question."
                )

            return OrchestratorResult(reply=reply, debug=debug)
        except Exception as exc:
            debug["status"] = "exception"
            debug["error"] = str(exc)
            return OrchestratorResult(
                reply=(
                    "The orchestrator could not complete this request. "
                    "Falling back to the baseline retrieval flow is recommended."
                ),
                debug=debug,
            )


def _load_all_vector_stores(index_dir: Path) -> dict[str, Any]:
    """Load all FAISS indexes under index_dir for local testing."""
    stores: dict[str, Any] = {}
    if not index_dir.exists():
        return stores

    for path in sorted(index_dir.iterdir()):
        if not path.is_dir():
            continue
        try:
            stores[path.name] = load_vector_store(index_dir, path.name)
        except Exception:
            continue
    return stores


def run_test_routine() -> None:
    """Run a quick manual test suite for InsightForgeOrchestrator."""
    project_root = Path(__file__).resolve().parent
    index_dir = project_root / "indexes"

    vector_stores = _load_all_vector_stores(index_dir)
    source_names = sorted(vector_stores.keys())

    def _route_all_sources(query: str, names: list[str]) -> tuple[list[str], dict[str, Any]]:
        return (
            names,
            {
                "query": query,
                "available_sources": names,
                "selected_sources": names,
                "raw_router_content": "",
                "router_status": "test_route_all_sources",
            },
        )

    orchestrator = InsightForgeOrchestrator(
        route_sources=_route_all_sources,
        search_sources=lambda _query, _stores: [],
        build_reply=lambda _query, _results, _sources: "",
    )

    test_queries = [
        "Show monthly sales trends by region.",
        "What business model innovation themes appear in the documents?",
        "Correlate customer satisfaction from sales data with knowledge-base recommendations.",
        "Who won the 1998 FIFA World Cup?",
    ]

    print("InsightForgeOrchestrator test routine")
    print(f"Index directory: {index_dir}")
    print(f"Vector stores loaded: {len(vector_stores)}")
    print(f"LangGraph available: {orchestrator._langgraph_available}")
    print("-" * 80)

    for idx, query in enumerate(test_queries, start=1):
        result = orchestrator.execute_query(query, source_names, vector_stores)
        requirement = str(result.debug.get("resource_requirement", "unknown"))
        agent_used = str(result.debug.get("agent_used", "unknown"))
        status = str(result.debug.get("status", "unknown"))
        selected_count = len(result.debug.get("selected_sources", []))

        print(f"{idx:>3}. Query: {query}")
        print(f"    requirement={requirement}  agent={agent_used}  status={status}  selected_sources={selected_count}")
        print(f"    Response: {result.reply}\n")


if __name__ == "__main__":
    run_test_routine()
