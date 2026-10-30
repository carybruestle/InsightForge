from __future__ import annotations

import importlib
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module-level eager background load
# ---------------------------------------------------------------------------
# FAISS's first call to faiss.read_index triggers slow C++ hardware detection
# on Windows.  We kick off the load in a daemon thread the moment this module
# is imported so the cost is hidden behind other startup work (dashboard
# rendering, AutoGen init, etc.).  KnowledgeBaseAgent._load_all_vector_stores()
# simply waits for this future when it needs the stores.

_INDEX_DIR = Path(__file__).resolve().parent / "indexes"
_STORE_FUTURE: Future | None = None


def _background_load(index_dir: Path) -> dict[str, Any]:
    """Load all FAISS stores and wire in the real embeddings client."""
    stores: dict[str, Any] = {}
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.embeddings import Embeddings
    except ImportError:
        return stores

    index_dirs = [p for p in sorted(index_dir.iterdir()) if p.is_dir()]
    if not index_dirs:
        return stores

    class _StubEmbeddings(Embeddings):
        """Embeddings placeholder used only while indexes are being loaded."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    stub = _StubEmbeddings()

    def _load_one(path: Path):
        return path.name, FAISS.load_local(
            str(path), stub, allow_dangerous_deserialization=True
        )

    with ThreadPoolExecutor(max_workers=len(index_dirs)) as pool:
        for fut in {pool.submit(_load_one, p): p for p in index_dirs}:
            try:
                name, store = fut.result()
                stores[name] = store
            except Exception:
                continue

    # Wire in the real embeddings client once.
    try:
        from embeddings import get_openai_embeddings
        real_embed = get_openai_embeddings()
        for store in stores.values():
            store.embedding_function = real_embed
    except Exception:
        pass

    return stores


if _INDEX_DIR.exists():
    _bg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kb_preload")
    _STORE_FUTURE = _bg_executor.submit(_background_load, _INDEX_DIR)
    _bg_executor.shutdown(wait=False)  # daemon thread; don't block process exit


class KnowledgeBaseAgent:
    """Answers knowledge-base questions using FAISS retrieval and AutoGen."""

    SYSTEM_PROMPT = (
        "You are a knowledge base interpreter. You answer user questions using only the provided KB chunks. "
        "Do not invent facts not present in the context. "
        "If the context is insufficient, clearly state what is missing. "
        "Provide concise, business-focused answers grounded in the knowledge base."
    )

    def __init__(self, index_dir: Path | None = None) -> None:
        """Initialize the agent with optional index directory path."""
        if index_dir is None:
            index_dir = Path(__file__).resolve().parent / "indexes"
        self.index_dir = index_dir
        self._vector_stores: dict[str, Any] | None = None

    def _load_all_vector_stores(self) -> dict[str, Any]:
        """Return stores from the module-level background future, or load now."""
        if _STORE_FUTURE is not None:
            return _STORE_FUTURE.result()  # waits only if still loading
        return _background_load(self.index_dir)


    def _retrieve_top_chunks(self, query: str, k: int = 5) -> list[dict[str, str]]:
        """Retrieve top-k chunks globally ranked across all KB indexes."""
        if self._vector_stores is None:
            self._vector_stores = self._load_all_vector_stores()

        scored_chunks: list[tuple[float, dict[str, str]]] = []
        for source_name, store in self._vector_stores.items():
            try:
                docs_with_scores = store.similarity_search_with_score(query, k=k)
                for doc, score in docs_with_scores:
                    content = (getattr(doc, "page_content", "") or "").strip()
                    if not content:
                        continue
                    scored_chunks.append(
                        (
                            float(score),
                            {
                                "source": source_name,
                                "content": content,
                            },
                        )
                    )
            except Exception:
                # Fallback path for stores that do not expose score APIs.
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
                            {
                                "source": source_name,
                                "content": content,
                            },
                        )
                    )

        scored_chunks.sort(key=lambda item: item[0])
        return [chunk for _, chunk in scored_chunks[:k]]

    def respond(self, query: str) -> str:
        """Return an answer for query using KB chunks and AutoGen."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "KnowledgeBaseAgent is unavailable because OPENAI_API_KEY is not set."

        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

        chunks = self._retrieve_top_chunks(query, k=5)
        if not chunks:
            return "KnowledgeBaseAgent found no relevant chunks in the knowledge base. Please try a different query."

        source_names = sorted({chunk["source"] for chunk in chunks})
        print(f"[KnowledgeBaseAgent] Retrieved sources: {', '.join(source_names)}")

        chunks_text = "\n".join(
            [chunk["content"] for chunk in chunks]
        )

        try:
            autogen_module = importlib.import_module("autogen")
            assistant_cls = getattr(autogen_module, "AssistantAgent")
            user_proxy_cls = getattr(autogen_module, "UserProxyAgent")

            config_item: dict[str, Any] = {"model": model, "api_key": api_key}
            if base_url:
                config_item["base_url"] = base_url

            assistant = assistant_cls(
                name="KnowledgeBaseAgent",
                llm_config={"config_list": [config_item], "temperature": 0},
                system_message=self.SYSTEM_PROMPT,
            )
            user_proxy = user_proxy_cls(
                name="kb_user",
                human_input_mode="NEVER",
                code_execution_config=False,
                max_consecutive_auto_reply=0,
            )

            prompt = (
                f"User query: {query}\n\n"
                "Knowledge base chunks (top 5 by relevance):\n"
                f"{chunks_text}\n\n"
                "Answer the query using these chunks only. Do not mention source names or document titles."
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

            return "KnowledgeBaseAgent could not generate a response from the KB chunks."
        except Exception as exc:
            return f"KnowledgeBaseAgent failed to run: {exc}"


def run_test_routine() -> None:
    """Run a quick manual test suite for KnowledgeBaseAgent."""
    test_queries = [
        "What are the key business model innovation themes?",
        "What insights about time-series data prediction are available?",
        "How does retail sales analysis approach customer trends?",
        "What strategic recommendations are discussed for improving retail performance?",
        "Which themes are emphasized around IoT-driven forecasting?",
        "What limitations or caveats are mentioned in the indexed papers?",
        "Summarize recurring ideas across the available business documents.",
    ]

    agent = KnowledgeBaseAgent()
    index_dir = agent.index_dir

    # Count index folders without loading (avoids triggering embeddings API prematurely).
    index_count = sum(1 for p in index_dir.iterdir() if p.is_dir()) if index_dir.exists() else 0

    print("KnowledgeBaseAgent test routine")
    print(f"Using index directory: {index_dir}")
    print(f"Indexes found: {index_count}")
    print("Loading vector stores...", flush=True)

    # Pre-warm once so queries share a single loaded store set.
    agent._vector_stores = agent._load_all_vector_stores()
    print(f"Loaded {len(agent._vector_stores)} vector stores.")
    print("-" * 80)

    for idx, query in enumerate(test_queries, start=1):
        response = agent.respond(query)
        print(f"{idx:>3}. {query}")
        print(f"    Response: {response}\n")


if __name__ == "__main__":
    run_test_routine()
