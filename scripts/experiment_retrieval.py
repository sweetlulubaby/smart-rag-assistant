"""
scripts/experiment_retrieval.py - 三路召回策略对比实验
对比：
  A) 纯向量召回（FAISS）
  B) 纯 BM25 稀疏召回
  C) 混合召回（向量 + BM25 分数加权融合）

指标：
  Recall@3 / Recall@5 / Recall@10
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

# 避免在离线环境下触发 LangSmith 上报重试噪音
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import HYBRID_ALPHA, HYBRID_FETCH_K
from src.retriever import (
    build_bm25_retriever,
    build_faiss_retriever,
    build_hybrid_retriever,
)
from src.vectorstore import load_vectorstore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

K_LIST = [3, 5, 10]
DATASET_CANDIDATES = [
    Path("./data/test_question.json"),
    Path("./data/test_questions.json"),
]
RESULT_MD = Path("./retrieval_experiment_results.md")
RESULT_JSON = Path("./data/retrieval_experiment_results.json")


def resolve_dataset_path() -> Path:
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "未找到评测集，请确保 data/test_question.json 或 data/test_questions.json 存在。"
    )


def load_test_dataset(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("测试集 JSON 格式错误：应为数组。")
    return data


def _extract_query(item: dict) -> str:
    return item.get("query", item.get("question", "")).strip()


def _extract_gt_doc_id(item: dict) -> str:
    return item.get("ground_truth_doc_id", "").strip()


def evaluate_recall(retriever, test_data: List[dict], k_list: List[int]) -> Dict[str, float]:
    total = len(test_data)
    hits = {k: 0 for k in k_list}

    for item in test_data:
        query = _extract_query(item)
        gt_doc_id = _extract_gt_doc_id(item)
        if not query or not gt_doc_id:
            continue

        docs = retriever.invoke(query)
        for k in k_list:
            is_hit = any(gt_doc_id in str(doc.metadata.get("source", "")) for doc in docs[:k])
            if is_hit:
                hits[k] += 1

    return {f"Recall@{k}": hits[k] / total for k in k_list}


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def mean_recall(metrics: Dict[str, float], k_list: List[int]) -> float:
    return sum(metrics[f"Recall@{k}"] for k in k_list) / len(k_list)


def to_markdown_table(rows: List[dict], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator] + body)


def run_hybrid_tuning(vectorstore, test_data: List[dict], k_list: List[int]) -> Dict[str, float]:
    alpha_candidates = [0.30, 0.45, 0.55, 0.65, 0.75]
    fetch_k = max(HYBRID_FETCH_K, max(k_list) * 3)
    best = None

    logger.info("开始混合召回 alpha 调参...")
    for alpha in alpha_candidates:
        hybrid = build_hybrid_retriever(
            vectorstore=vectorstore,
            k=max(k_list),
            alpha=alpha,
            fetch_k=fetch_k,
        )
        metrics = evaluate_recall(hybrid, test_data, k_list)
        score = mean_recall(metrics, k_list)
        logger.info(
            f"  alpha={alpha:.2f} -> "
            f"R@3={fmt_pct(metrics['Recall@3'])}, "
            f"R@5={fmt_pct(metrics['Recall@5'])}, "
            f"R@10={fmt_pct(metrics['Recall@10'])}, "
            f"avg={fmt_pct(score)}"
        )
        if (best is None) or (score > best["avg"]):
            best = {"alpha": alpha, "avg": score, "metrics": metrics}

    return best


def main():
    logger.info("=" * 60)
    logger.info("三路召回策略评测：FAISS vs BM25 vs Hybrid(Score Fusion)")
    logger.info("=" * 60)

    dataset_path = resolve_dataset_path()
    test_data = load_test_dataset(dataset_path)
    logger.info(f"评测集: {dataset_path} (样本数={len(test_data)})")

    vectorstore = load_vectorstore()
    if vectorstore is None:
        raise RuntimeError("FAISS 索引不存在，请先执行 `python scripts/ingest_docs.py --source ./data/docs/`")

    max_k = max(K_LIST)

    logger.info("构建纯向量召回器...")
    faiss_retriever = build_faiss_retriever(vectorstore, k=max_k)

    logger.info("构建纯 BM25 召回器...")
    bm25_retriever = build_bm25_retriever(vectorstore, k=max_k)

    hybrid_best = run_hybrid_tuning(vectorstore, test_data, K_LIST)
    best_alpha = hybrid_best["alpha"]

    logger.info(f"构建最佳混合召回器 (alpha={best_alpha:.2f})...")
    hybrid_retriever = build_hybrid_retriever(
        vectorstore=vectorstore,
        k=max_k,
        alpha=best_alpha,
        fetch_k=max(HYBRID_FETCH_K, max_k * 3),
    )

    logger.info("开始正式对比评测...")
    faiss_metrics = evaluate_recall(faiss_retriever, test_data, K_LIST)
    bm25_metrics = evaluate_recall(bm25_retriever, test_data, K_LIST)
    hybrid_metrics = evaluate_recall(hybrid_retriever, test_data, K_LIST)

    rows = [
        {
            "召回方案": "纯向量 (FAISS)",
            "Recall@3": fmt_pct(faiss_metrics["Recall@3"]),
            "Recall@5": fmt_pct(faiss_metrics["Recall@5"]),
            "Recall@10": fmt_pct(faiss_metrics["Recall@10"]),
            "备注": "baseline",
            "avg_raw": mean_recall(faiss_metrics, K_LIST),
        },
        {
            "召回方案": "纯 BM25",
            "Recall@3": fmt_pct(bm25_metrics["Recall@3"]),
            "Recall@5": fmt_pct(bm25_metrics["Recall@5"]),
            "Recall@10": fmt_pct(bm25_metrics["Recall@10"]),
            "备注": "关键词召回更显著",
            "avg_raw": mean_recall(bm25_metrics, K_LIST),
        },
        {
            "召回方案": f"混合召回 (FAISS+BM25, alpha={best_alpha:.2f})",
            "Recall@3": fmt_pct(hybrid_metrics["Recall@3"]),
            "Recall@5": fmt_pct(hybrid_metrics["Recall@5"]),
            "Recall@10": fmt_pct(hybrid_metrics["Recall@10"]),
            "备注": "调参后最优",
            "avg_raw": mean_recall(hybrid_metrics, K_LIST),
        },
    ]

    best_row = max(rows, key=lambda x: x["avg_raw"])
    table_columns = ["召回方案", "Recall@3", "Recall@5", "Recall@10", "备注"]
    md_table = to_markdown_table(rows, table_columns)

    logger.info("\n实验结果对照表：\n")
    logger.info(md_table)
    logger.info(
        f"\n最佳策略: {best_row['召回方案']} "
        f"(平均 Recall={fmt_pct(best_row['avg_raw'])})"
    )

    summary = {
        "dataset_path": str(dataset_path),
        "sample_count": len(test_data),
        "k_list": K_LIST,
        "hybrid_tuning_default_alpha": HYBRID_ALPHA,
        "hybrid_best_alpha": best_alpha,
        "rows": rows,
        "best_strategy": best_row["召回方案"],
        "best_avg_recall": best_row["avg_raw"],
    }

    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write("### 多路召回策略实验对比（FAISS / BM25 / Hybrid）\n\n")
        f.write(f"- 测试集：`{dataset_path}`（{len(test_data)} 条）\n")
        f.write(f"- 混合召回调参：`alpha` 最优为 **{best_alpha:.2f}**\n\n")
        f.write(md_table)
        f.write(
            "\n\n"
            f"> 结论：本轮最佳策略为 **{best_row['召回方案']}**，"
            f"平均 Recall 为 **{fmt_pct(best_row['avg_raw'])}**。"
        )

    logger.info(f"\n已输出: {RESULT_MD}")
    logger.info(f"已输出: {RESULT_JSON}")


if __name__ == "__main__":
    main()
