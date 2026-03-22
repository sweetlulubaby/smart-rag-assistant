"""
src/robust_retriever.py - 第六步：空召回兜底与热点问题缓存
"""
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document
from pydantic.v1 import PrivateAttr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RETRIEVAL_SLOW_THRESHOLD_MS
from src.reranker import has_low_rerank_confidence

logger = logging.getLogger(__name__)

SYNONYM_MAP = {
    "装机量": ["装车量", "销量", "出货量"],
    "营收": ["营业收入", "收入"],
    "净利": ["净利润", "归母净利润"],
    "毛利率": ["毛利", "盈利能力"],
    "市占率": ["市场份额", "份额"],
    "跌幅": ["跌幅居前", "下跌幅度", "降幅"],
    "涨幅": ["涨幅居前", "上涨幅度", "增幅"],
    "同比": ["同比增长", "同比变动"],
    "环比": ["环比增长", "环比变动"],
}


def _doc_signature(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return f"{source}|{page}|{digest}"


def generate_query_rewrites(query: str) -> List[str]:
    rewrites: List[str] = []
    base = (query or "").strip()
    if not base:
        return rewrites

    for term, replacements in SYNONYM_MAP.items():
        if term in base:
            for replacement in replacements:
                rewrites.append(base.replace(term, replacement))

    # 常见金融问答里，额外补一个“关键词式”短查询，提升稀疏/重排回收率
    compact = base.replace("请问", "").replace("一下", "").replace("有哪些", "")
    compact = compact.replace("是什么", "").replace("多少", "").strip("，。？！ ")
    if compact and compact != base:
        rewrites.append(compact)

    unique: List[str] = []
    seen = set()
    for item in rewrites:
        normalized = item.strip()
        if normalized and normalized not in seen and normalized != base:
            seen.add(normalized)
            unique.append(normalized)
    return unique[:4]


def needs_query_retry(docs: Sequence[Document]) -> bool:
    if not docs:
        return True
    if has_low_rerank_confidence(docs):
        return True
    return False


def _annotate_docs(docs: Sequence[Document], query_variant: str, cache_hit: bool) -> List[Document]:
    annotated: List[Document] = []
    for doc in docs:
        metadata = dict(doc.metadata)
        metadata["query_variant"] = query_variant
        metadata["cache_hit"] = cache_hit
        annotated.append(Document(page_content=doc.page_content, metadata=metadata))
    return annotated


def merge_documents(candidates: Sequence[Document], top_n: int) -> List[Document]:
    merged: List[Document] = []
    seen = set()
    for doc in candidates:
        sig = _doc_signature(doc)
        if sig in seen:
            continue
        seen.add(sig)
        merged.append(doc)

    if any(doc.metadata.get("rerank_score") is not None for doc in merged):
        merged.sort(key=lambda d: float(d.metadata.get("rerank_score", 0.0)), reverse=True)

    return merged[:top_n]


class RobustRetriever(BaseRetriever):
    """
    包装现有 retriever：
    - 精确 query 命中缓存时直接返回
    - 空召回或低置信度时自动做同义词改写重试
    """
    base_retriever: Any
    k: int
    enable_query_rewrite: bool = True
    enable_cache: bool = True

    _cache: Dict[str, List[Document]] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []

        if self.enable_cache and normalized_query in self._cache:
            return _annotate_docs(self._cache[normalized_query], normalized_query, cache_hit=True)

        start = time.perf_counter()
        docs = list(self.base_retriever.invoke(normalized_query))
        latency_ms = (time.perf_counter() - start) * 1000.0

        selected = docs[: self.k]
        if self.enable_query_rewrite and needs_query_retry(selected):
            retry_candidates = list(selected)
            for rewrite in generate_query_rewrites(normalized_query):
                rewrite_docs = list(self.base_retriever.invoke(rewrite))
                retry_candidates.extend(_annotate_docs(rewrite_docs, rewrite, cache_hit=False))
            selected = merge_documents(retry_candidates, self.k)
        else:
            selected = _annotate_docs(selected, normalized_query, cache_hit=False)

        if latency_ms >= RETRIEVAL_SLOW_THRESHOLD_MS:
            logger.info("slow_retrieval: query=%s latency_ms=%.1f", normalized_query, latency_ms)

        if self.enable_cache:
            self._cache[normalized_query] = [
                Document(page_content=doc.page_content, metadata=dict(doc.metadata))
                for doc in selected
            ]

        return selected


def build_robust_retriever(
    base_retriever: Any,
    k: int,
    enable_query_rewrite: bool = True,
    enable_cache: bool = True,
) -> RobustRetriever:
    return RobustRetriever(
        base_retriever=base_retriever,
        k=k,
        enable_query_rewrite=enable_query_rewrite,
        enable_cache=enable_cache,
    )
