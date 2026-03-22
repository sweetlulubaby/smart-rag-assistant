"""
src/answering.py - 第六步：带引用回答、低置信度兜底与答案后校验
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence

from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel

from src.reranker import has_low_rerank_confidence

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
UNCERTAIN_ANSWER = (
    "我不确定。根据现有资料，暂时没有足够证据支持一个明确结论。"
    "你可以补充更具体的时间范围、主体名称或章节关键词，我再继续核对。"
)


def _llm_text(llm: BaseLanguageModel, prompt: str) -> str:
    resp = llm.invoke(prompt)
    if hasattr(resp, "content"):
        return str(resp.content)
    return str(resp)


def format_docs_for_prompt(docs: Sequence[Document], max_chars: int = 12000) -> str:
    parts: List[str] = []
    total = 0
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知文档")
        source_name = Path(str(source)).name if source else "未知文档"
        page = doc.metadata.get("page", "")
        page_info = ""
        if page != "":
            try:
                page_info = f" 第{int(page) + 1}页"
            except Exception:
                page_info = ""
        score = doc.metadata.get("rerank_score", None)
        score_info = f" rerank={float(score):.4f}" if score is not None else ""
        chunk = f"[{i}] {source_name}{page_info}{score_info}\n{doc.page_content}".strip()
        parts.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    return "\n\n---\n\n".join(parts)


def detect_answer_issues(answer: str, doc_count: int) -> List[str]:
    issues: List[str] = []
    text = (answer or "").strip()
    if not text:
        return ["empty_answer"]

    citation_indices = [int(m.group(1)) for m in CITATION_PATTERN.finditer(text)]
    if citation_indices:
        if any(idx < 1 or idx > doc_count for idx in citation_indices):
            issues.append("invalid_citation_index")
    elif "我不确定" not in text and "资料不足" not in text:
        issues.append("missing_citation")

    if len(text) < 8:
        issues.append("too_short")
    return issues


def should_fallback_to_uncertain(docs: Sequence[Document], answer: str, issues: Sequence[str]) -> bool:
    if not docs:
        return True
    if has_low_rerank_confidence(docs):
        return True
    if "empty_answer" in issues:
        return True
    if "invalid_citation_index" in issues:
        return True
    if "missing_citation" in issues and len(answer.strip()) < 80:
        return True
    return False


def generate_grounded_answer(
    llm: BaseLanguageModel,
    question: str,
    docs: Sequence[Document],
) -> Dict[str, object]:
    """
    生成带引用回答，并在发现空答/无引用/引用越界时进行一次修复。
    """
    context = format_docs_for_prompt(docs)
    if not context.strip():
        return {
            "answer": UNCERTAIN_ANSWER,
            "issues": ["no_context"],
            "used_repair": False,
        }

    prompt = (
        "你是一个严谨的中文金融文档问答助手。请只根据给定资料回答，不要补充常识猜测。\n"
        "输出要求：\n"
        "1. 每一个关键结论后都要附证据编号，如 [1]、[2]。\n"
        "2. 如果资料不足以支持明确结论，直接回答“我不确定”，并说明缺少什么信息。\n"
        "3. 不要编造数字、日期、公司名称或因果关系。\n"
        "4. 回答尽量简洁，优先给结论，再给必要说明。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}\n"
    )
    try:
        answer = _llm_text(llm, prompt).strip()
    except Exception as exc:
        logger.warning("answer_generation_error: %s", exc)
        return {
            "answer": UNCERTAIN_ANSWER,
            "issues": ["generation_error"],
            "used_repair": False,
        }
    issues = detect_answer_issues(answer, len(docs))

    used_repair = False
    if issues:
        repair_prompt = (
            "你现在要对一个候选回答做事实后校验和修复。请只依据资料片段保留有证据支持的内容。\n"
            "修复规则：\n"
            "1. 删除资料中没有证据的表述。\n"
            "2. 每一个保留的关键结论后必须附 [编号]。\n"
            "3. 如果资料不足，输出：我不确定。根据现有资料，暂时没有足够证据支持明确结论。\n"
            "4. 不要解释你的检查过程，只输出修复后的最终回答。\n\n"
            f"资料：\n{context}\n\n"
            f"问题：{question}\n\n"
            f"候选回答：\n{answer}\n\n"
            f"发现的问题：{', '.join(issues)}\n"
        )
        try:
            answer = _llm_text(llm, repair_prompt).strip()
            issues = detect_answer_issues(answer, len(docs))
            used_repair = True
        except Exception as exc:
            logger.warning("answer_repair_error: %s", exc)
            issues = ["repair_error"]

    if should_fallback_to_uncertain(docs, answer, issues):
        logger.info("answer_fallback: issues=%s", issues)
        answer = UNCERTAIN_ANSWER

    return {
        "answer": answer,
        "issues": issues,
        "used_repair": used_repair,
    }
