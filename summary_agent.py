from __future__ import annotations

import importlib
import os
import re
from typing import Any, Sequence


class SummaryAgent:
    """Summarizes up to three ordered query/response pairs into one or two sentences."""

    SYSTEM_PROMPT = (
        "You summarize conversations for memory compression. "
        "Write exactly one or two professional sentences. "
        "Capture the user's goal, key actions taken, and current status. "
        "Do not include bullet points, JSON, or markdown. "
        "Avoid implementation details unless they are essential to status."
    )

    def _format_conversation(self, conversation: Sequence[tuple[str, str]]) -> str:
        """Render a short, ordered conversation into prompt-friendly text."""
        pairs = list(conversation[:3])
        if not pairs:
            return ""

        rendered_pairs = []
        for index, pair in enumerate(pairs, start=1):
            if len(pair) != 2:
                continue
            query, response = pair
            query_text = str(query).strip()
            response_text = str(response).strip()
            if not query_text and not response_text:
                continue
            rendered_pairs.append(
                f"Turn {index}: Query: {query_text}\nTurn {index}: Response: {response_text}"
            )

        return "\n\n".join(rendered_pairs)

    def _heuristic_summary(self, conversation_text: str) -> str:
        """Fallback summarizer when LLM is unavailable."""
        cleaned = re.sub(r"\s+", " ", conversation_text).strip()
        if not cleaned:
            return "No conversation content was provided to summarize."

        # Split into rough sentence chunks and keep only informative lines.
        chunks = re.split(r"(?<=[.!?])\s+", cleaned)
        informative = [
            chunk.strip()
            for chunk in chunks
            if chunk.strip() and len(chunk.strip()) > 25
        ]

        if not informative:
            return "The conversation included limited detail and did not provide enough information for a strong summary."

        if len(informative) == 1:
            return informative[0]

        first = informative[0]
        last = informative[-1]
        if first == last:
            return first

        return f"{first} {last}"

    def summarize(self, conversation: Sequence[tuple[str, str]]) -> str:
        """Return a one- or two-sentence summary of up to three conversation pairs."""
        conversation_text = self._format_conversation(conversation)
        if not conversation_text.strip():
            return "No conversation content was provided to summarize."

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._heuristic_summary(conversation_text)

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
                name="SummaryAgent",
                llm_config={"config_list": [config_item], "temperature": 0},
                system_message=self.SYSTEM_PROMPT,
            )
            user_proxy = user_proxy_cls(
                name="summary_user",
                human_input_mode="NEVER",
                code_execution_config=False,
                max_consecutive_auto_reply=0,
            )

            prompt = (
                "Summarize this conversation into one or two sentences. "
                "The conversation is provided as up to three ordered query/response pairs. "
                "Focus on user intent, what was done, and current state.\n\n"
                f"Conversation:\n{conversation_text}"
            )

            try:
                user_proxy.initiate_chat(assistant, message=prompt, silent=True)
            except TypeError:
                user_proxy.initiate_chat(assistant, message=prompt)

            last_message = assistant.last_message() if hasattr(assistant, "last_message") else None
            if isinstance(last_message, dict):
                content = str(last_message.get("content", "")).strip()
                if content:
                    # Guardrail: trim to at most two sentences if model drifts.
                    sentences = re.split(r"(?<=[.!?])\s+", content)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    if len(sentences) > 2:
                        return " ".join(sentences[:2])
                    return content

            return self._heuristic_summary(conversation_text)
        except Exception:
            return self._heuristic_summary(conversation_text)


def run_test_routine() -> None:
    """Run a quick manual test suite for SummaryAgent."""
    agent = SummaryAgent()

    test_conversations = [
        (
            (
                "How does customer satisfaction relate to sales performance?",
                "Sales performance appears to rise when customer satisfaction scores are higher.",
            ),
            (
                "What analysis should we use to confirm that relationship?",
                "A regression model and trend review would help quantify the connection.",
            ),
            (
                "How should we present the result?",
                "A simple visualization would make the relationship easier to explain to stakeholders.",
            ),
        ),

    ]

    print("SummaryAgent test routine")
    print("-" * 80)
    for idx, convo in enumerate(test_conversations, start=1):
        summary = agent.summarize(convo)
        print(f"{idx:>3}. Conversation: {convo}")
        print(f"    Summary: {summary}\n")


if __name__ == "__main__":
    run_test_routine()
