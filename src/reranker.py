"""
src/reranker.py - 第六步：接入 Cross-Encoder Reranker
使用 FlagEmbedding 的 bge-reranker-v2-m3 对召回候选进行重排。
"""
import logging
from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RERANK_FETCH_K,
    RERANK_LOW_SCORE_THRESHOLD,
    RERANK_MODEL_NAME,
    RERANK_TOP_N,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker_model():
    from FlagEmbedding import FlagReranker

    logger.info("加载 Reranker 模型: %s", RERANK_MODEL_NAME)
    return FlagReranker(RERANK_MODEL_NAME, use_fp16=False)


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_n: int = RERANK_TOP_N,
) -> List[Document]:
    """
    对候选文档进行重排，并将 rerank_score 写入 metadata。
    """
    docs = list(documents)
    if not docs:
        return []

    reranker = get_reranker_model()
    pairs: List[Tuple[str, str]] = [(query, doc.page_content) for doc in docs]
    scores = reranker.compute_score(pairs)

    if isinstance(scores, (int, float)):
        scores = [float(scores)]

    ranked = []
    for idx, (doc, score) in enumerate(zip(docs, scores), start=1):
        metadata = dict(doc.metadata)
        metadata["rerank_score"] = float(score)
        metadata["retrieval_rank_before_rerank"] = idx
        ranked.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    ranked.sort(key=lambda item: item.metadata.get("rerank_score", 0.0), reverse=True)
    return ranked[:top_n]


class RerankRetriever(BaseRetriever):
    """
    对任意基础 retriever 的输出结果进行 rerank。
    """
    base_retriever: Any
    fetch_k: int = RERANK_FETCH_K
    top_n: int = RERANK_TOP_N

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        return rerank_documents(query, docs[: self.fetch_k], top_n=self.top_n)


def build_rerank_retriever(
    base_retriever: BaseRetriever,
    fetch_k: int = RERANK_FETCH_K,
    top_n: int = RERANK_TOP_N,
) -> RerankRetriever:
    return RerankRetriever(
        base_retriever=base_retriever,
        fetch_k=fetch_k,
        top_n=top_n,
    )


def has_low_rerank_confidence(docs: Sequence[Document]) -> bool:
    """
    仅在存在 rerank_score 时使用该信号，否则返回 False。
    """
    if not docs:
        return True
    top_score = docs[0].metadata.get("rerank_score")
    if top_score is None:
        return False
    try:
        return float(top_score) < RERANK_LOW_SCORE_THRESHOLD
    except Exception:
        return False
