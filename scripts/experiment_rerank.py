"""
scripts/experiment_rerank.py - 升级六：Reranker 对比实验
对比：
1. 纯向量直接 Top5
2. 纯向量 Top20 -> Rerank -> Top5
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reranker import build_rerank_retriever
from src.retriever import build_faiss_retriever
from src.vectorstore import load_vectorstore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_CANDIDATES = [
    Path("./data/test_question.json"),
    Path("./data/test_questions.json"),
]
RESULT_MD = Path("./rerank_experiment_results.md")
RESULT_JSON = Path("./data/rerank_experiment_results.json")


def resolve_dataset_path() -> Path:
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("未找到评测集，请确保测试集文件存在。")


def load_test_dataset(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("测试集 JSON 格式错误：应为数组。")
    return data


def to_markdown_table(rows: List[dict], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator] + body)


def evaluate_top5(retriever, test_data: List[dict]) -> Dict[str, float]:
    total = len(test_data)
    recall_hits = 0
    precision_hits = 0
    elapsed = 0.0

    for item in test_data:
        query = item.get("query", item.get("question", "")).strip()
        gt_doc_id = item.get("ground_truth_doc_id", "").strip()
        if not query or not gt_doc_id:
            continue

        start = time.perf_counter()
        docs = retriever.invoke(query)
        elapsed += time.perf_counter() - start

        matched = 0
        for doc in docs[:5]:
            source = str(doc.metadata.get("source", ""))
            if gt_doc_id in source:
                matched += 1

        if matched > 0:
            recall_hits += 1
        precision_hits += matched

    return {
        "Recall@5": recall_hits / total,
        "Precision@5": precision_hits / (total * 5),
        "AvgLatencyMs": elapsed * 1000.0 / total,
    }


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt_ms(value: float) -> str:
    return f"{value:.1f} ms"


def main():
    logger.info("=" * 60)
    logger.info("升级六：Reranker 对比实验")
    logger.info("=" * 60)

    dataset_path = resolve_dataset_path()
    test_data = load_test_dataset(dataset_path)
    logger.info("评测集: %s (样本数=%s)", dataset_path, len(test_data))

    vectorstore = load_vectorstore()
    if vectorstore is None:
        raise RuntimeError("FAISS 索引不存在，请先完成文档入库。")

    direct_top5 = build_faiss_retriever(vectorstore, k=5)
    rerank_top5 = build_rerank_retriever(
        build_faiss_retriever(vectorstore, k=20),
        fetch_k=20,
        top_n=5,
    )

    logger.info("开始评测：直接 Top5")
    direct_metrics = evaluate_top5(direct_top5, test_data)
    logger.info("开始评测：Top20 -> Rerank -> Top5")
    rerank_metrics = evaluate_top5(rerank_top5, test_data)

    rows = [
        {
            "方案": "直接 Top5 (FAISS)",
            "Recall@5": fmt_pct(direct_metrics["Recall@5"]),
            "Precision@5": fmt_pct(direct_metrics["Precision@5"]),
            "平均耗时": fmt_ms(direct_metrics["AvgLatencyMs"]),
            "备注": "无重排",
        },
        {
            "方案": "Top20 -> Rerank -> Top5",
            "Recall@5": fmt_pct(rerank_metrics["Recall@5"]),
            "Precision@5": fmt_pct(rerank_metrics["Precision@5"]),
            "平均耗时": fmt_ms(rerank_metrics["AvgLatencyMs"]),
            "备注": "bge-reranker-v2-m3",
        },
    ]

    columns = ["方案", "Recall@5", "Precision@5", "平均耗时", "备注"]
    md_table = to_markdown_table(rows, columns)
    logger.info("\n%s", md_table)

    delta_recall = rerank_metrics["Recall@5"] - direct_metrics["Recall@5"]
    delta_precision = rerank_metrics["Precision@5"] - direct_metrics["Precision@5"]

    payload = {
        "dataset_path": str(dataset_path),
        "sample_count": len(test_data),
        "rows": rows,
        "delta_recall_at_5": delta_recall,
        "delta_precision_at_5": delta_precision,
    }

    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write("### 升级六：Reranker 对比实验\n\n")
        f.write(f"- 测试集：`{dataset_path}`（{len(test_data)} 条）\n")
        f.write("- 召回链路：纯向量召回\n")
        f.write("- 对比方案：直接 Top5 vs Top20 后经 `bge-reranker-v2-m3` 重排取 Top5\n\n")
        f.write(md_table)
        f.write(
            "\n\n"
            f"> Recall@5 变化：{fmt_pct(delta_recall)}；"
            f"Precision@5 变化：{fmt_pct(delta_precision)}。"
        )

    logger.info("已输出: %s", RESULT_MD)
    logger.info("已输出: %s", RESULT_JSON)


if __name__ == "__main__":
    main()
