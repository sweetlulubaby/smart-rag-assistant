from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain.schema import BaseRetriever, Document
from langchain_core.language_models import BaseLanguageModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from src.answering import UNCERTAIN_ANSWER, generate_grounded_answer

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    chat_history: List[Dict[str, Any]]
    action: Literal["rag", "direct", "clarify"]
    docs: List[Document]
    answer: str
    sources: List[Dict[str, Any]]


def _tail_chat_history(chat_history: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    if not chat_history:
        return []
    tail = chat_history[-limit:]
    compact: List[Dict[str, Any]] = []
    for item in tail:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            compact.append({"role": role, "content": content.strip()})
    return compact


def _llm_text(llm: BaseLanguageModel, prompt: str) -> str:
    resp = llm.invoke(prompt)
    if hasattr(resp, "content"):
        return str(resp.content)
    return str(resp)


def _format_context(docs: List[Document], max_chars: int = 12_000) -> str:
    parts: List[str] = []
    total = 0
    for i, doc in enumerate(docs, 1):
        source_raw = doc.metadata.get("source", "")
        source = Path(source_raw).name if isinstance(source_raw, str) and source_raw else "unknown"
        page = doc.metadata.get("page", None)
        page_info = ""
        if page is not None:
            try:
                page_info = f" p.{int(page) + 1}"
            except Exception:
                page_info = ""
        chunk = f"[{i}] {source}{page_info}\n{doc.page_content}".strip()
        if not chunk:
            continue
        parts.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    return "\n\n---\n\n".join(parts)


def _build_sources(docs: List[Document], limit: int = 4) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs[:limit], start=1):
        source_raw = doc.metadata.get("source", "")
        source = Path(source_raw).name if isinstance(source_raw, str) and source_raw else "unknown"
        page = doc.metadata.get("page", 0)
        sources.append({"ref": idx, "source": source, "page": page})
    return sources


@dataclass(frozen=True)
class LangGraphRAGAgent:
    app: Any

    def invoke(self, question: str, chat_history: Optional[List[Dict[str, Any]]] = None, thread_id: str = "default"):
        state: AgentState = {
            "question": question,
            "chat_history": _tail_chat_history(chat_history or []),
        }
        return self.app.invoke(state, config={"configurable": {"thread_id": thread_id}})


def build_langgraph_rag_agent(
    retriever: BaseRetriever,
    llm: BaseLanguageModel,
    *,
    enable_router: bool = True,
    enable_clarify: bool = True,
) -> LangGraphRAGAgent:
    """
    A small LangGraph agent for this repo:
    - route (rag/direct/clarify)
    - retrieve (tool call)
    - generate (answer with citations)
    - direct/clarify fallbacks
    """

    def route(state: AgentState) -> Dict[str, Any]:
        question = (state.get("question") or "").strip()
        history = state.get("chat_history") or []

        if not enable_router:
            return {"action": "rag"}

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        prompt = (
            "You are a router for a PDF-based RAG assistant.\n"
            "Decide the best next action for answering the user's question.\n"
            "Actions:\n"
            "- rag: use the document retriever and answer grounded in documents\n"
            "- direct: answer without retrieval (small talk / general, not document-specific)\n"
            "- clarify: ask a short clarifying question when the user's intent is ambiguous\n\n"
            "Return ONLY a JSON object like: {\"action\": \"rag\"}\n\n"
            f"Chat history (may be empty):\n{history_text}\n\n"
            f"User question:\n{question}\n"
        )

        try:
            text = _llm_text(llm, prompt)
            data = json.loads(text.strip())
            action = data.get("action")
            if action in ("rag", "direct", "clarify"):
                if action == "clarify" and not enable_clarify:
                    return {"action": "rag"}
                return {"action": action}
        except Exception as e:
            logger.info("router_fallback: %s", e)

        heuristic = "rag"
        if any(k in question.lower() for k in ["hi", "hello", "hey", "你好", "在吗"]):
            heuristic = "direct"
        if enable_clarify and len(question) < 6:
            heuristic = "clarify"
        return {"action": heuristic}

    def retrieve(state: AgentState) -> Dict[str, Any]:
        question = (state.get("question") or "").strip()
        docs = retriever.invoke(question)
        return {"docs": docs, "sources": _build_sources(docs)}

    def generate(state: AgentState) -> Dict[str, Any]:
        question = (state.get("question") or "").strip()
        docs = state.get("docs") or []
        if not docs:
            return {
                "answer": UNCERTAIN_ANSWER,
                "sources": [],
            }
        result = generate_grounded_answer(llm, question, docs)
        return {"answer": result.get("answer", UNCERTAIN_ANSWER)}

    def direct(state: AgentState) -> Dict[str, Any]:
        question = (state.get("question") or "").strip()
        prompt = (
            "你是一个中文助手。用户的问题不一定需要查资料。\n"
            "请给出简洁、可执行的建议。如果问题涉及用户的本地文档内容，请提示用户改用资料检索模式。\n\n"
            f"问题：{question}\n"
        )
        return {"answer": _llm_text(llm, prompt).strip(), "sources": []}

    def clarify(state: AgentState) -> Dict[str, Any]:
        question = (state.get("question") or "").strip()
        prompt = (
            "你是一个中文助手。用户问题可能不完整。\n"
            "请只提出1个最关键的澄清问题（不要给出最终答案），帮助你继续检索或回答。\n\n"
            f"用户问题：{question}\n"
        )
        return {"answer": _llm_text(llm, prompt).strip(), "sources": []}

    graph = StateGraph(AgentState)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("direct", direct)
    graph.add_node("clarify", clarify)

    graph.set_entry_point("route")

    def _route_edge(state: AgentState) -> str:
        return state.get("action", "rag")

    graph.add_conditional_edges(
        "route",
        _route_edge,
        {
            "rag": "retrieve",
            "direct": "direct",
            "clarify": "clarify",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("direct", END)
    graph.add_edge("clarify", END)

    app = graph.compile(checkpointer=MemorySaver())
    return LangGraphRAGAgent(app=app)
