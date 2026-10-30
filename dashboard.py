from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any

LOG_FILE = Path(__file__).resolve().parent / "insightforge.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger(__name__)


def _log_debug(label: str, data: Any) -> None:
    """Write a labelled JSON snapshot to the log file."""
    _log.debug("%s: %s", label, json.dumps(data, indent=2, ensure_ascii=True, default=str))

import pandas as pd
import streamlit as st

from embeddings import load_vector_store
from embeddings import retrieve
from agents_orchestrator import InsightForgeOrchestrator
from summary_agent import SummaryAgent

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "Data" / "sales_data.csv"
INDEX_DIR = PROJECT_ROOT / "indexes"


@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
    """Load sales data and derive demographic/time helper columns."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    df["Customer_Gender"] = df["Customer_Gender"].fillna("Unknown")
    df["Age_Group"] = pd.cut(
        df["Customer_Age"],
        bins=[0, 24, 34, 44, 54, 64, 200],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        include_lowest=True,
        right=True,
    ).astype("string").fillna("Unknown")

    return df


def filter_data(df: pd.DataFrame, ui: Any) -> pd.DataFrame:
    """Apply filters for time period, region, product, and demographics."""
    ui.markdown("**Filters**")
    ui.caption("Compact sidebar controls")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    ui.markdown("<span style='font-size:0.78rem;font-weight:600;'>Time Period</span>", unsafe_allow_html=True)
    date_range = ui.date_input(
        "Time Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed",
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = min_date
        end_date = max_date

    regions = sorted(df["Region"].dropna().unique().tolist())
    products = sorted(df["Product"].dropna().unique().tolist())
    genders = sorted(df["Customer_Gender"].dropna().unique().tolist())
    age_groups = sorted(df["Age_Group"].dropna().unique().tolist())

    row1_left, row1_right = ui.columns(2, gap="small")
    with row1_left:
        st.caption("Region")
        selected_regions = ui.multiselect("Region", regions, default=regions, label_visibility="collapsed")
    with row1_right:
        st.caption("Product")
        selected_products = ui.multiselect("Product", products, default=products, label_visibility="collapsed")

    row2_left, row2_right = ui.columns(2, gap="small")
    with row2_left:
        st.caption("Gender")
        selected_genders = ui.multiselect("Customer Gender", genders, default=genders, label_visibility="collapsed")
    with row2_right:
        st.caption("Age")
        selected_age_groups = ui.multiselect("Age Group", age_groups, default=age_groups, label_visibility="collapsed")

    mask = (
        (df["Date"].dt.date >= start_date)
        & (df["Date"].dt.date <= end_date)
        & (df["Region"].isin(selected_regions))
        & (df["Product"].isin(selected_products))
        & (df["Customer_Gender"].isin(selected_genders))
        & (df["Age_Group"].isin(selected_age_groups))
    )

    return df.loc[mask].copy()


def build_time_summary(filtered_df: pd.DataFrame, ui: Any) -> pd.DataFrame:
    """Aggregate filtered sales by selected time grain."""
    time_grain = ui.selectbox("Group sales by", ["Day", "Month", "Quarter", "Year"], index=1)

    freq_map = {
        "Day": "D",
        "Month": "M",
        "Quarter": "Q",
        "Year": "Y",
    }
    freq = freq_map[time_grain]

    with_period = filtered_df.assign(
        Period=filtered_df["Date"].dt.to_period(freq).astype(str),
    )
    grouped = (
        with_period.groupby("Period", as_index=False)
        .agg(
            **{
                "Total Sales": ("Sales", "sum"),
                "Sales Std Dev": ("Sales", "std"),
            }
        )
        .fillna({"Sales Std Dev": 0.0})
        .sort_values("Period")
    )
    return grouped


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_all_vector_stores(index_dir: Path) -> dict[str, Any]:
    """Load every persisted FAISS index from index_dir."""
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


def search_vector_stores(
    query: str,
    stores: dict[str, Any],
    k_per_store: int = 2,
    max_results: int = 6,
) -> list[dict[str, str]]:
    """Run similarity search across selected stores and collect text snippets."""
    results: list[dict[str, str]] = []
    for source_name, store in stores.items():
        try:
            docs = retrieve(store, query, k=k_per_store)
        except Exception:
            continue

        for doc in docs:
            content = (getattr(doc, "page_content", "") or "").strip()
            if not content:
                continue
            results.append({"source": source_name, "content": content})
            if len(results) >= max_results:
                return results

    return results


def select_relevant_sources_with_autogen(query: str, source_names: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Use an AutoGen agent to choose which PDF sources to search."""
    debug_info: dict[str, Any] = {
        "query": query,
        "available_sources": source_names,
        "selected_sources": [],
        "raw_router_content": "",
        "router_status": "fallback",
    }

    if not source_names:
        debug_info["router_status"] = "no_sources"
        return [], debug_info

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        debug_info["router_status"] = "missing_api_key"
        debug_info["selected_sources"] = source_names
        return source_names, debug_info

    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

    try:
        autogen_module = importlib.import_module("autogen")
        assistant_cls = getattr(autogen_module, "AssistantAgent")
        user_proxy_cls = getattr(autogen_module, "UserProxyAgent")

        config_item: dict[str, Any] = {"model": model, "api_key": api_key}
        if base_url:
            config_item["base_url"] = base_url

        llm_config: dict[str, Any] = {
            "config_list": [config_item],
            "temperature": 0,
        }

        assistant = assistant_cls(
            name="pdf_router",
            llm_config=llm_config,
            system_message=(
                "You are a routing agent. Select which PDF sources are most relevant to a query. "
                "Return ONLY a JSON object with a single key 'sources' containing an array of source names."
            ),
        )
        user_proxy = user_proxy_cls(
            name="router_user",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
        )

        prompt = (
            "Available sources:\n"
            f"{json.dumps(source_names)}\n\n"
            f"User query: {query}\n\n"
            "Return JSON like: {\"sources\": [\"source1\", \"source2\"]}."
        )
        user_proxy.initiate_chat(assistant, message=prompt)
        last_message = assistant.last_message() if hasattr(assistant, "last_message") else None
        if not isinstance(last_message, dict):
            debug_info["router_status"] = "invalid_router_response"
            debug_info["selected_sources"] = source_names
            return source_names, debug_info

        content = str(last_message.get("content", "")).strip()
        debug_info["raw_router_content"] = content
        parsed = json.loads(content)
        selected = parsed.get("sources", []) if isinstance(parsed, dict) else []
        selected_set = {str(item) for item in selected}
        routed = [name for name in source_names if name in selected_set]
        final_sources = routed or source_names
        debug_info["selected_sources"] = final_sources
        debug_info["router_status"] = "ok"
        return final_sources, debug_info
    except Exception:
        # Fall back safely to searching all sources if AutoGen is unavailable or misconfigured.
        debug_info["router_status"] = "exception"
        debug_info["selected_sources"] = source_names
        return source_names, debug_info


def build_chat_reply(
    query: str,
    search_results: list[dict[str, str]],
    selected_sources: list[str],
) -> str:
    """Create a concise chat response from retrieved chunks."""
    if not search_results:
        return (
            "I could not find relevant context in the loaded vector stores. "
            "Try a more specific question or confirm indexes are built."
        )

    excerpts: list[str] = []
    for idx, item in enumerate(search_results[:4], start=1):
        snippet = " ".join(item["content"].split())
        if len(snippet) > 280:
            snippet = f"{snippet[:280]}..."
        excerpts.append(f"{idx}. ({item['source']}) {snippet}")

    joined = "\n".join(excerpts)
    routed_sources = ", ".join(selected_sources)
    return (
        f"AutoGen routed this question to: {routed_sources}.\n\n"
        f"For your question \"{query}\", I found these relevant excerpts:\n\n"
        f"{joined}\n\n"
        "If you want, ask a follow-up and I can narrow to one source document."
    )


def _summarize_turn_pairs(turn_pairs: list[tuple[str, str]]) -> str:
    """Summarize any number of ordered query/response pairs into a short memory."""
    if not turn_pairs:
        return ""

    agent = SummaryAgent()

    def _compress(pairs: list[tuple[str, str]]) -> str:
        if not pairs:
            return ""
        if len(pairs) <= 3:
            return agent.summarize(pairs)

        reduced_pairs: list[tuple[str, str]] = []
        for chunk_index, start in enumerate(range(0, len(pairs), 3), start=1):
            chunk = pairs[start:start + 3]
            chunk_summary = agent.summarize(chunk)
            reduced_pairs.append((f"Earlier conversation summary {chunk_index}", chunk_summary))

        return _compress(reduced_pairs)

    return _compress(turn_pairs)


def _sync_chat_summary_state() -> None:
    """Keep the rolling summary aligned with the stored chat history."""
    chat_history = st.session_state.get("chat_history", [])
    summarized_pair_count = st.session_state.get("chat_summary_pair_count", 0)
    target_pair_count = max(0, len(chat_history) - 1)

    if target_pair_count <= summarized_pair_count:
        return

    previous_summary = st.session_state.get("chat_summary", "")
    new_pairs = [
        (entry.get("query", ""), entry.get("response", ""))
        for entry in chat_history[summarized_pair_count:target_pair_count]
        if isinstance(entry, dict)
    ]

    if not new_pairs:
        st.session_state.chat_summary_pair_count = target_pair_count
        return

    summary_source: list[tuple[str, str]] = []
    if previous_summary:
        summary_source.append(("Earlier conversation summary", previous_summary))
    summary_source.extend(new_pairs)

    st.session_state.chat_summary = _summarize_turn_pairs(summary_source)
    st.session_state.chat_summary_pair_count = target_pair_count


def _build_contextual_query(
    user_query: str,
    chat_history: list[dict[str, str]],
    older_summary: str,
) -> str:
    """Build an effective query that carries short conversation memory."""
    context_blocks: list[str] = []

    summary_text = older_summary.strip()
    if summary_text:
        context_blocks.append(f"Earlier conversation summary:\n{summary_text}")

    recent_turns = [entry for entry in chat_history[-3:] if isinstance(entry, dict)]
    if recent_turns:
        turn_lines: list[str] = []
        for idx, turn in enumerate(recent_turns, start=1):
            query_text = str(turn.get("query", "")).strip()
            response_text = str(turn.get("response", "")).strip()
            if not query_text and not response_text:
                continue
            turn_lines.append(
                f"Recent turn {idx} - Query: {query_text}\n"
                f"Recent turn {idx} - Response: {response_text}"
            )
        if turn_lines:
            context_blocks.append("Recent conversation turns (oldest to newest):\n" + "\n\n".join(turn_lines))

    if not context_blocks:
        return user_query

    return (
        "Use the provided conversation context to resolve references in the latest user question. "
        "If the latest question remains ambiguous after using context, ask a targeted clarifying question.\n\n"
        f"{'\n\n'.join(context_blocks)}\n\n"
        f"Latest user question:\n{user_query}"
    )


def execute_with_orchestrator(
    query: str,
    source_names: list[str],
    vector_stores: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Run orchestrator path and return reply, router debug, and orchestrator debug."""
    orchestrator = InsightForgeOrchestrator(
        route_sources=select_relevant_sources_with_autogen,
        search_sources=search_vector_stores,
        build_reply=build_chat_reply,
    )
    result = orchestrator.execute_query(
        query=query,
        source_names=source_names,
        vector_stores=vector_stores,
    )
    router_debug = result.debug.get("router_debug")
    if not isinstance(router_debug, dict):
        router_debug = {
            "query": query,
            "available_sources": source_names,
            "selected_sources": source_names,
            "raw_router_content": "",
            "router_status": "orchestrator_no_router_debug",
        }
    return result.reply, router_debug, result.debug


def execute_with_fallback_flow(
    query: str,
    source_names: list[str],
    vector_stores: dict[str, Any],
    display_query: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run baseline flow used before orchestrator integration."""
    selected_sources, router_debug = select_relevant_sources_with_autogen(query, source_names)
    active_stores = {name: vector_stores[name] for name in selected_sources}
    search_results = search_vector_stores(query, active_stores)
    reply = build_chat_reply(display_query or query, search_results, selected_sources)
    return reply, router_debug


def render_chat_pane(vector_stores: dict[str, Any]) -> None:
    """Render a conversation pane that retrieves context from FAISS vector stores."""
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] [data-testid="stMultiSelect"] {
            margin-bottom: 0.15rem;
        }
        div[data-testid="stSidebar"] label {
            font-size: 0.78rem !important;
            margin-bottom: 0.15rem !important;
        }
        div[data-testid="stSidebar"] [data-baseweb="select"] > div {
            min-height: 2.0rem !important;
            font-size: 0.78rem !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stDateInput"] input {
            font-size: 0.78rem !important;
            padding-top: 0.2rem !important;
            padding-bottom: 0.2rem !important;
        }
        .st-key-chat-pane {
            background-color: #eef4fb;
            border-radius: 0.75rem;
            border: 1px solid #c3d9f0;
            padding: 1rem 0.85rem 0.85rem 0.85rem;
        }
        .chat-pane-banner {
            background-color: #e8f1fb;
            border: 1px solid #c3d9f0;
            border-radius: 0.6rem;
            color: #1f3552;
            padding: 0.7rem 0.85rem;
            margin: 0 0 0.85rem 0;
            font-size: 0.9rem;
            font-weight: 600;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] div {
            color: #1a1a1a !important;
        }
        .st-key-chat-pane [data-testid="stButton"] > button {
            background-color: #d9e9fb;
            border: 1px solid #9bbce0;
            color: #163354;
            font-size: 0.78rem;
            padding: 0.3rem 0.55rem;
        }
        .st-key-chat-pane [data-testid="stButton"] > button:hover,
        .st-key-chat-pane [data-testid="stButton"] > button:focus {
            background-color: #c8def7;
            border-color: #7ca8d6;
            color: #102b48;
        }
        .st-key-chat-pane [data-testid="stButton"] > button:disabled {
            background-color: #e9eff6;
            border-color: #d0dceb;
            color: #7a8da4;
            opacity: 1;
        }
        .st-key-chat-history-pane {
            max-height: 28rem;
            overflow-y: auto;
            border: 1px solid #c3d9f0;
            border-radius: 0.6rem;
            background-color: #f7fbff;
            padding: 0.4rem 0.55rem;
            margin-top: 0.6rem;
        }
        .chat-loading {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            background-color: #eaf3ff;
            border: 1px solid #b9d1ec;
            color: #163354;
            border-radius: 0.55rem;
            padding: 0.45rem 0.65rem;
            margin-top: 0.45rem;
            font-size: 0.86rem;
            font-weight: 600;
        }
        .chat-loading-dot {
            width: 0.72rem;
            height: 0.72rem;
            border-radius: 50%;
            background-color: #2d7cc0;
            box-shadow: 0 0 0 0 rgba(45, 124, 192, 0.55);
            animation: chatPulse 1.1s ease-in-out infinite;
            flex: 0 0 auto;
        }
        @keyframes chatPulse {
            0% {
                transform: scale(0.88);
                box-shadow: 0 0 0 0 rgba(45, 124, 192, 0.55);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 0.44rem rgba(45, 124, 192, 0);
            }
            100% {
                transform: scale(0.88);
                box-shadow: 0 0 0 0 rgba(45, 124, 192, 0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="chat-pane"):
        if not vector_stores:
            st.info("No vector stores found. Build indexes first by running embeddings.py.")
            return

        source_names = sorted(vector_stores.keys())

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "chat_summary" not in st.session_state:
            st.session_state.chat_summary = ""
        if "chat_summary_pair_count" not in st.session_state:
            st.session_state.chat_summary_pair_count = 0

        banner_col, clear_col = st.columns([0.78, 0.22], gap="small")
        with banner_col:
            st.markdown(
                """
                <div class="chat-pane-banner">
                    Ask a business question to generate an AI-assisted analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with clear_col:
            clear_clicked = st.button(
                "Clear chat",
                key="clear_chat_button",
                help="Clear conversation history and summary.",
                use_container_width=True,
                disabled=not (st.session_state.chat_history or st.session_state.chat_summary),
            )
        if clear_clicked:
            st.session_state.chat_history = []
            st.session_state.chat_summary = ""
            st.session_state.chat_summary_pair_count = 0
            st.rerun()

        user_prompt = st.chat_input(
            "Type your question",
            key="top_chat_input",
        )

        if user_prompt and user_prompt.strip():
            contextual_query = _build_contextual_query(
                user_query=user_prompt,
                chat_history=st.session_state.chat_history,
                older_summary=st.session_state.chat_summary,
            )

            loading_placeholder = st.empty()
            loading_placeholder.markdown(
                """
                <div class="chat-loading">
                    <span class="chat-loading-dot"></span>
                    Analyzing your question, selecting data sources, and generating an answer...
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                use_orchestrator = os.getenv("INSIGHTFORGE_USE_ORCHESTRATOR", "1") != "0"

                if use_orchestrator:
                    try:
                        reply, router_debug, orchestrator_debug = execute_with_orchestrator(
                            contextual_query,
                            source_names,
                            vector_stores,
                        )
                        _log_debug("router_debug", router_debug)
                        _log_debug("orchestrator_debug", orchestrator_debug)
                    except Exception:
                        reply, router_debug = execute_with_fallback_flow(
                            contextual_query,
                            source_names,
                            vector_stores,
                            display_query=user_prompt,
                        )
                        _log_debug("router_debug", router_debug)
                        _log_debug("orchestrator_debug", {"status": "fallback_after_exception", "execution_mode": "fallback", "langgraph_available": False})
                else:
                    reply, router_debug = execute_with_fallback_flow(
                        contextual_query,
                        source_names,
                        vector_stores,
                        display_query=user_prompt,
                    )
                    _log_debug("router_debug", router_debug)
                    _log_debug("orchestrator_debug", {"status": "disabled_by_env", "execution_mode": "fallback", "langgraph_available": False})
            finally:
                loading_placeholder.empty()

            st.session_state.chat_history.append({"query": user_prompt, "response": reply})
            _sync_chat_summary_state()
            st.rerun()

        _sync_chat_summary_state()

        with st.container(key="chat-history-pane"):
            latest_turn = st.session_state.chat_history[-1] if st.session_state.chat_history else None
            if isinstance(latest_turn, dict):
                query_text = latest_turn.get("query", "")
                response_text = latest_turn.get("response", "")
                with st.chat_message("user"):
                    st.write(query_text)
                with st.chat_message("assistant"):
                    st.write(response_text)

            if st.session_state.chat_summary:
                st.markdown("**Earlier conversation**")
                st.info(st.session_state.chat_summary)


def main() -> None:
    st.set_page_config(page_title="InsightForge", layout="wide")
    st.markdown(
        """
        <style>
        section.main > div.block-container {
            padding-top: 0.7rem;
            padding-bottom: 1.0rem;
        }
        section.main h1 {
            margin-top: 0;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("InsightForge")
    st.caption("AI Assisted Business Intelligence")

    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}")
        return

    left_pane, right_pane = st.columns([1.95, 1.45], gap="medium")

    with left_pane:
        df = load_data(DATA_FILE)
        st.sidebar.header("Filters")
        filtered_df = filter_data(df, st.sidebar)

        if filtered_df.empty:
            st.warning("No records match the selected filters.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{len(filtered_df):,}")
            col2.metric("Total Sales", f"{filtered_df['Sales'].sum():,.0f}")
            col3.metric("Avg Satisfaction", f"{filtered_df['Customer_Satisfaction'].mean():.2f}")
            col4.metric("Sales Std Dev", f"{filtered_df['Sales'].std(ddof=1):,.2f}")

            breakdown = (
                filtered_df.groupby(["Region", "Product", "Customer_Gender", "Age_Group"], as_index=False)
                .agg(
                    Total_Sales=("Sales", "sum"),
                    Sales_Std_Dev=("Sales", "std"),
                    Avg_Satisfaction=("Customer_Satisfaction", "mean"),
                    Records=("Sales", "count"),
                )
                .fillna({"Sales_Std_Dev": 0.0})
                .sort_values(["Total_Sales", "Records"], ascending=[False, False])
            )

            time_summary = build_time_summary(filtered_df, st.sidebar)
            trend_tab, breakdown_tab, raw_tab = st.tabs(["Trend", "Breakdown", "Raw Data"])

            with trend_tab:
                st.line_chart(time_summary.set_index("Period")["Total Sales"])
                st.dataframe(time_summary, width='stretch', height=260)

            with breakdown_tab:
                st.dataframe(breakdown, width='stretch', height=430)

            with raw_tab:
                st.dataframe(filtered_df.sort_values("Date"), width='stretch', height=430)

    with right_pane:
        vector_stores = load_all_vector_stores(INDEX_DIR)
        render_chat_pane(vector_stores)


if __name__ == "__main__":
    main()
