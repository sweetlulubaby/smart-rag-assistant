"""
src/evaluation.py - 第五步：自动化量化评估 (Ragas)
使用 Ragas 框架对 RAG 系统进行自动化评分，计算：
  - Faithfulness（忠实度）：答案有没有胡编乱造
  - Answer Relevancy（答案相关性）：有没有答非所问
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TEST_DATASET_PATH,
    EVAL_RESULTS_PATH,
    RETRIEVAL_MODE,
    get_llm,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_dataset(dataset_path: str = TEST_DATASET_PATH) -> List[Dict[str, str]]:
    """
    加载测试数据集（JSON 格式）。
    
    数据格式：
    [
        {
            "question": "问题",
            "ground_truth": "标准答案（可选）"
        },
        ...
    ]
    
    Args:
        dataset_path: JSON 文件路径
    
    Returns:
        问题列表
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"加载测试数据集: {len(data)} 个问题")
    return data


def generate_rag_answers(
    test_data: List[Dict[str, str]],
    rag_chain,
    retriever,
) -> List[Dict[str, Any]]:
    """
    用 RAG 链对测试数据集中的每个问题生成答案，并收集检索到的文档。
    
    Args:
        test_data: 测试数据列表
        rag_chain: LCEL RAG 链
        retriever: 检索器（用于获取 contexts）
    
    Returns:
        包含 question/answer/contexts/ground_truth 的字典列表
    """
    results = []
    
    for i, item in enumerate(test_data):
        # 兼容新旧格式
        question = item.get("query", item.get("question", ""))
        ground_truth = item.get("answer", item.get("ground_truth", ""))
        q_type = item.get("type", "未知")
        doc_id = item.get("ground_truth_doc_id", "")
        
        logger.info(f"  [{i+1}/{len(test_data)}] 生成答案: {question[:50]}...")
        
        # 生成答案（非流式）
        answer = rag_chain.invoke(question)
        
        # 获取检索到的文档（用于 Ragas 上下文评估）
        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]
        
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "type": q_type,
            "ground_truth_doc_id": doc_id,
        })
    
    return results


def run_ragas_evaluation(
    results: List[Dict[str, Any]],
    metrics: List = None,
) -> pd.DataFrame:
    """
    使用 Ragas 计算评估指标。
    
    Args:
        results: generate_rag_answers 的输出
        metrics: 评估指标列表（默认包含忠实度和答案相关性）
    
    Returns:
        评估结果 DataFrame
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset
    
    if metrics is None:
        metrics = [faithfulness, answer_relevancy]
    
    logger.info("准备 Ragas 评估数据集...")
    
    # 构建 Ragas 需要的 Dataset 格式
    eval_data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
    }
    
    # 如果有 ground_truth，加入以支持更多指标
    if all(r.get("ground_truth") for r in results):
        eval_data["ground_truth"] = [r["ground_truth"] for r in results]
    
    dataset = Dataset.from_dict(eval_data)
    
    logger.info(f"开始 Ragas 评估（指标: {[m.name for m in metrics]}）...")
    logger.info("注意：Ragas 评估需要调用 LLM API，可能需要几分钟")
    
    score = evaluate(dataset=dataset, metrics=metrics)
    
    results_df = score.to_pandas()
    
    # 将额外信息加回 DataFrame 以便于分析
    if "type" in results[0]:
        results_df["type"] = [r["type"] for r in results]
    if "ground_truth_doc_id" in results[0]:
        results_df["ground_truth_doc_id"] = [r.get("ground_truth_doc_id", "") for r in results]

    logger.info(f"\n{'='*50}")
    logger.info("Ragas 总体评估结果：")
    logger.info(f"  Faithfulness（忠实度）:       {results_df['faithfulness'].mean():.4f}")
    logger.info(f"  Answer Relevancy（答案相关性）: {results_df['answer_relevancy'].mean():.4f}")
    
    if "type" in results_df.columns:
        logger.info("\n按 问题类型(type) 分类统计：")
        grouped = results_df.groupby("type")[["faithfulness", "answer_relevancy"]].mean()
        logger.info(f"\n{grouped}")
        
    logger.info(f"{'='*50}\n")
    
    return results_df


def save_evaluation_results(results_df: pd.DataFrame, output_path: str = EVAL_RESULTS_PATH):
    """
    保存评估结果到 CSV 文件。
    
    Args:
        results_df: Ragas 评估结果 DataFrame
        output_path: 输出 CSV 路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"✅ 评估结果已保存至: {output_path}")


def run_full_evaluation(
    test_dataset_path: str = TEST_DATASET_PATH,
    output_path: str = EVAL_RESULTS_PATH,
    use_multi_query: bool = True,
    retrieval_mode: str = RETRIEVAL_MODE,
):
    """
    执行完整的评估流程（入口函数）。
    
    Args:
        test_dataset_path: 测试数据集路径
        output_path: 结果输出路径
        use_multi_query: 是否使用 MultiQueryRetriever
    """
    from src.vectorstore import get_vectorstore
    from src.retriever import get_retriever, get_simple_retriever
    from src.rag_chain import get_rag_chain
    
    logger.info("=" * 60)
    logger.info("开始 RAG 系统自动化评估")
    logger.info("=" * 60)
    
    # 初始化组件
    llm = get_llm(streaming=False)
    vectorstore = get_vectorstore()
    
    if use_multi_query:
        retriever = get_retriever(vectorstore, llm, mode=retrieval_mode)
        logger.info(f"使用 MultiQueryRetriever（多查询扩展, 策略={retrieval_mode}）")
    else:
        retriever = get_simple_retriever(vectorstore, mode=retrieval_mode)
        logger.info(f"使用基础检索器（策略={retrieval_mode}）")
    
    rag_chain = get_rag_chain(retriever, llm)
    
    # 加载测试数据
    test_data = load_test_dataset(test_dataset_path)
    
    # 生成答案
    logger.info("\n步骤 1/2: 生成 RAG 答案...")
    rag_results = generate_rag_answers(test_data, rag_chain, retriever)
    
    # Ragas 评估
    logger.info("\n步骤 2/2: Ragas 评估...")
    results_df = run_ragas_evaluation(rag_results)
    
    # 保存结果
    save_evaluation_results(results_df, output_path)
    
    return results_df


if __name__ == "__main__":
    run_full_evaluation()
