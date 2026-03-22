"""
src/retriever.py - 第三步：统一检索策略工厂
支持三种基础召回策略：
1) 纯向量（FAISS）
2) 纯稀疏（BM25）
3) 混合召回（向量 + BM25 分数加权融合）

并保留 MultiQueryRetriever 作为可选“外层增强器”。
"""
import hashlib
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import BaseRetriever, Document
from langchain_core.language_models import BaseLanguageModel
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    HYBRID_ALPHA,
    HYBRID_FETCH_K,
    HOT_QUERY_CACHE_ENABLED,
    QUERY_REWRITE_ENABLED,
    RERANK_ENABLED,
    RERANK_FETCH_K,
    RERANK_TOP_N,
    RETRIEVAL_MODE,
    RETRIEVER_K,
)
from src.robust_retriever import build_robust_retriever
from src.reranker import build_rerank_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize_zh_en(text: str) -> List[str]:
    """
    中文+英文混合分词：
    - 英文/数字按词切分
    - 中文按单字切分（无需额外依赖）
    """
    if not text:
        return []
    tokens = TOKEN_PATTERN.findall(text.lower())
    return tokens if tokens else list(text.lower())


def _doc_signature(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return f"{source}|{page}|{digest}"


def _normalize_dense_distance(distance: float) -> float:
    # FAISS 距离越小越相关，这里转成越大越相关的相似度分值
    d = max(float(distance), 0.0)
    return 1.0 / (1.0 + d)


def _minmax_scale(values: List[float]) -> List[float]:
    if not values:
        return values
    v_min = min(values)
    v_max = max(values)
    if v_max - v_min < 1e-12:
        return [1.0 if v_max > 0 else 0.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def get_index_documents(vectorstore: FAISS) -> List[Document]:
    """
    从 FAISS docstore 还原全部 chunk 文档，用于 BM25 与混合检索。
    """
    docstore_dict = getattr(vectorstore.docstore, "_dict", {})
    if not docstore_dict:
        return []
    # 按 key 排序，确保可复现
    ordered = sorted(docstore_dict.items(), key=lambda kv: str(kv[0]))
    return [doc for _, doc in ordered]


class WeightedHybridRetriever(BaseRetriever):
    """
    分数加权混合检索器：
    final_score = alpha * dense_score + (1 - alpha) * bm25_score
    """
    vectorstore: FAISS
    bm25_retriever: BM25Retriever
    documents: List[Document]
    doc_id_by_signature: Dict[str, int]
    alpha: float = HYBRID_ALPHA
    k: int = RETRIEVER_K
    fetch_k: int = HYBRID_FETCH_K
    preprocess_func: Callable[[str], List[str]] = tokenize_zh_en

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        if not self.documents:
            return []

        dense_pairs: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(
            query=query, k=self.fetch_k
        )

        dense_scores_by_id: Dict[int, float] = {}
        for doc, distance in dense_pairs:
            sig = _doc_signature(doc)
            doc_id = self.doc_id_by_signature.get(sig)
            if doc_id is None:
                continue
            dense_score = _normalize_dense_distance(distance)
            # 同一文档出现多次时保留最高分
            if doc_id not in dense_scores_by_id or dense_score > dense_scores_by_id[doc_id]:
                dense_scores_by_id[doc_id] = dense_score

        query_tokens = self.preprocess_func(query)
        bm25_scores = self.bm25_retriever.vectorizer.get_scores(query_tokens)
        bm25_scores = np.array(bm25_scores, dtype=float)
        top_bm25_ids = np.argsort(-bm25_scores)[: self.fetch_k].tolist()

        candidate_ids = set(dense_scores_by_id.keys()) | set(top_bm25_ids)
        if not candidate_ids:
            return []

        candidate_list = list(candidate_ids)
        dense_raw = [dense_scores_by_id.get(doc_id, 0.0) for doc_id in candidate_list]
        bm25_raw = [float(bm25_scores[doc_id]) for doc_id in candidate_list]

        dense_scaled = _minmax_scale(dense_raw)
        bm25_scaled = _minmax_scale(bm25_raw)

        combined = []
        alpha = min(max(self.alpha, 0.0), 1.0)
        for i, doc_id in enumerate(candidate_list):
            score = alpha * dense_scaled[i] + (1.0 - alpha) * bm25_scaled[i]
            combined.append((doc_id, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in combined[: self.k]]
        return [self.documents[doc_id] for doc_id in top_ids]


def build_faiss_retriever(vectorstore: FAISS, k: int = RETRIEVER_K) -> BaseRetriever:
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def build_bm25_retriever(
    vectorstore: FAISS,
    k: int = RETRIEVER_K,
    preprocess_func: Callable[[str], List[str]] = tokenize_zh_en,
) -> BM25Retriever:
    documents = get_index_documents(vectorstore)
    bm25 = BM25Retriever.from_documents(
        documents,
        preprocess_func=preprocess_func,
    )
    bm25.k = k
    return bm25


def build_hybrid_retriever(
    vectorstore: FAISS,
    k: int = RETRIEVER_K,
    alpha: float = HYBRID_ALPHA,
    fetch_k: int = HYBRID_FETCH_K,
    preprocess_func: Callable[[str], List[str]] = tokenize_zh_en,
) -> WeightedHybridRetriever:
    documents = get_index_documents(vectorstore)
    bm25 = BM25Retriever.from_documents(
        documents,
        preprocess_func=preprocess_func,
    )
    bm25.k = fetch_k

    doc_id_by_signature = {
        _doc_signature(doc): idx for idx, doc in enumerate(documents)
    }
    return WeightedHybridRetriever(
        vectorstore=vectorstore,
        bm25_retriever=bm25,
        documents=documents,
        doc_id_by_signature=doc_id_by_signature,
        alpha=alpha,
        k=k,
        fetch_k=fetch_k,
        preprocess_func=preprocess_func,
    )


def get_simple_retriever(
    vectorstore: FAISS,
    mode: str = RETRIEVAL_MODE,
    k: int = RETRIEVER_K,
    enable_rerank: bool = RERANK_ENABLED,
) -> BaseRetriever:
    """
    基础检索器工厂（无 MultiQuery）：
    - mode=faiss / bm25 / hybrid
    """
    normalized_mode = (mode or RETRIEVAL_MODE).strip().lower()

    base_k = max(k, RERANK_FETCH_K) if enable_rerank else k

    if normalized_mode == "faiss":
        logger.info(f"使用纯向量检索器 (FAISS, fetch={base_k}, final_top={k})")
        base_retriever = build_faiss_retriever(vectorstore, k=base_k)
    elif normalized_mode == "bm25":
        logger.info(f"使用纯稀疏检索器 (BM25, fetch={base_k}, final_top={k})")
        base_retriever = build_bm25_retriever(vectorstore, k=base_k)
    else:
        logger.info(
            "使用混合检索器 (Hybrid Score Fusion, "
            f"alpha={HYBRID_ALPHA:.2f}, fetch_k={HYBRID_FETCH_K}, final_top={k})"
        )
        base_retriever = build_hybrid_retriever(
            vectorstore,
            k=base_k,
            alpha=HYBRID_ALPHA,
            fetch_k=HYBRID_FETCH_K,
        )

    if enable_rerank:
        logger.info(
            "在基础检索器后启用 Reranker (fetch=%s -> top=%s)",
            base_k,
            min(k, RERANK_TOP_N if k == RETRIEVER_K else k),
        )
        base_retriever = build_rerank_retriever(
            base_retriever,
            fetch_k=base_k,
            top_n=k,
        )
    return build_robust_retriever(
        base_retriever,
        k=k,
        enable_query_rewrite=QUERY_REWRITE_ENABLED,
        enable_cache=HOT_QUERY_CACHE_ENABLED,
    )


def get_retriever(
    vectorstore: FAISS,
    llm: BaseLanguageModel,
    mode: str = RETRIEVAL_MODE,
    k: int = RETRIEVER_K,
    enable_rerank: bool = RERANK_ENABLED,
) -> BaseRetriever:
    """
    在基础检索器外包裹 MultiQuery 以提升召回。
    """
    base_retriever = get_simple_retriever(
        vectorstore,
        mode=mode,
        k=max(k, RERANK_FETCH_K) if enable_rerank else k,
        enable_rerank=False,
    )

    try:
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )
        logger.info(
            f"✅ MultiQueryRetriever 构建完成 "
            f"(策略={mode}, 每子查询返回 top-{k})"
        )
        if enable_rerank:
            logger.info(
                "在 MultiQuery 结果后启用 Reranker (fetch=%s -> top=%s)",
                max(k, RERANK_FETCH_K),
                k,
            )
            retriever = build_rerank_retriever(
                retriever,
                fetch_k=max(k, RERANK_FETCH_K),
                top_n=k,
            )
            return build_robust_retriever(
                retriever,
                k=k,
                enable_query_rewrite=QUERY_REWRITE_ENABLED,
                enable_cache=HOT_QUERY_CACHE_ENABLED,
            )
        return build_robust_retriever(
            retriever,
            k=k,
            enable_query_rewrite=QUERY_REWRITE_ENABLED,
            enable_cache=HOT_QUERY_CACHE_ENABLED,
        )
    except Exception as e:
        logger.warning(
            f"⚠️ MultiQueryRetriever 构建失败，回退基础检索器: {e}"
        )
        if enable_rerank:
            base_retriever = build_rerank_retriever(
                base_retriever,
                fetch_k=max(k, RERANK_FETCH_K),
                top_n=k,
            )
        return build_robust_retriever(
            base_retriever,
            k=k,
            enable_query_rewrite=QUERY_REWRITE_ENABLED,
            enable_cache=HOT_QUERY_CACHE_ENABLED,
        )


def format_retrieved_docs(docs: List[Document]) -> str:
    """
    将检索到的文档列表格式化为字符串，供 Prompt 使用。
    """
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "")
        page_info = f" (第{page + 1}页)" if page != "" else ""

        formatted_parts.append(
            f"【参考文档 {i}】来源: {source}{page_info}\n{doc.page_content}"
        )

    return "\n\n" + "\n\n".join(formatted_parts)
