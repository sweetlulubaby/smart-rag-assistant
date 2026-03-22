"""
scripts/generate_eval_dataset.py - 自动生成评测集
使用 LLM 从本地文档中自动生成评测集，包含：事实型、对比型、汇总型三类。
最终生成的数据格式为： (query, ground_truth_doc_id, answer)
"""
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_llm, TEST_DATASET_PATH
from src.pdf_parser import parse_pdf
from src.ingestion import chunk_documents


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class QAPair(BaseModel):
    query: str = Field(description="生成的问题")
    answer: str = Field(description="基于文档给出的标准答案")
    type: str = Field(description="问题类型，只能是：'事实型', '对比型', '汇总型'")

class QAOutput(BaseModel):
    qa_list: List[QAPair] = Field(description="生成的问答对列表")


FACTOID_PROMPT = """
你是一个金融与数据归纳专家。请基于以下文档片段，生成 2 个**事实型**问题及对应的标准答案。
事实型问题的核心是“答案确定性强”，例如“XX公司的营收是多少”、“某产品的定价是多少”。回答要求精准、简练。

文档内容：
{context}

要求输出格式严格遵守如下 JSON schema：
{format_instructions}
"""

COMPARISON_PROMPT = """
你是一个金融与数据研报专家。请基于以下几段文档内容，生成 1 到 2 个**对比型**问题及标准答案。
对比型问题要求跨越文档中的不同实体、时间点或维度进行对比，例如“A与B相比，谁的毛利率更高，具体数值是多少”。

文档内容：
{context}

要求输出格式严格遵守如下 JSON schema：
{format_instructions}
"""

SUMMARY_PROMPT = """
你是一个行业研究资深分析师。请基于以下几段文档内容，生成 1 到 2 个**汇总型**问题及标准答案。
汇总型问题要求对文档中的某一宏观主题进行归纳概括，例如“2025年新能源行业的主要政策变化有哪些”。

文档内容：
{context}

要求输出格式严格遵守如下 JSON schema：
{format_instructions}
"""


def extract_doc_id(doc_path: str) -> str:
    """提取 ground_truth_doc_id (即文件名)"""
    return Path(doc_path).name


def generate_qa_pairs(llm, prompt_template: str, context: str, parser: JsonOutputParser) -> List[Dict]:
    """调用 LLM 生成 QA 对"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    try:
        # LLM 调用
        prompt_val = prompt.format(context=context)
        output = llm.invoke(prompt_val)
        
        # 解析输出
        parsed_output = parser.parse(output.content)
        return parsed_output.get("qa_list", [])
    except Exception as e:
        logger.error(f"LLM 生成 QA 对失败: {e}")
        return []


def main():
    docs_dir = Path("./data/docs")
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("未找到任何 PDF 文件在 ./data/docs 目录下！")
        return
    
    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件。")
    
    # 为了生成 80-120 条数据，我们抽取前 15 个文件处理
    # 假设每个文件产生 6-8 个问题，15个文件刚好能达到 90-120 条
    target_files = pdf_files[:15]
    
    llm = get_llm(streaming=False)
    parser = JsonOutputParser(pydantic_object=QAOutput)
    
    all_dataset = []
    
    for file_path in target_files:
        doc_id = extract_doc_id(str(file_path))
        logger.info(f"正在处理文档: {doc_id} ...")
        
        try:
            # 解析与切块 (取前 10 页以加速处理并避免 token 超限)
            docs = parse_pdf(str(file_path))
            if not docs:
                continue
            
            docs = docs[:10]
            chunks = chunk_documents(docs)
            
            if len(chunks) == 0:
                continue
            
            # 1. 事实型 (Factoid) - 随机抽取 2 个块，每个块生成 2 个问题 -> 共 4 个
            sampled_chunks = random.sample(chunks, min(2, len(chunks)))
            for chunk in sampled_chunks:
                qa_list = generate_qa_pairs(llm, FACTOID_PROMPT, chunk.page_content, parser)
                for qa in qa_list:
                    all_dataset.append({
                        "query": qa["query"],
                        "ground_truth_doc_id": doc_id,
                        "answer": qa["answer"],
                        "type": qa["type"]
                    })
            
            # 2. 对比型 (Comparison) - 抽取 2 到 3 个块拼接
            comp_chunks = random.sample(chunks, min(3, len(chunks)))
            comp_context = "\n\n---\n\n".join([c.page_content for c in comp_chunks])
            qa_list = generate_qa_pairs(llm, COMPARISON_PROMPT, comp_context, parser)
            for qa in qa_list:
                # 强行修正类型，避免 LLM 不听话
                all_dataset.append({
                    "query": qa["query"],
                    "ground_truth_doc_id": doc_id,
                    "answer": qa["answer"],
                    "type": "对比型"
                })
                
            # 3. 汇总型 (Summary) - 抽取 2 到 3 个块拼接
            summ_chunks = random.sample(chunks, min(3, len(chunks)))
            summ_context = "\n\n---\n\n".join([c.page_content for c in summ_chunks])
            qa_list = generate_qa_pairs(llm, SUMMARY_PROMPT, summ_context, parser)
            for qa in qa_list:
                all_dataset.append({
                    "query": qa["query"],
                    "ground_truth_doc_id": doc_id,
                    "answer": qa["answer"],
                    "type": "汇总型"
                })
                
            logger.info(f"  已从 {doc_id} 抽取并累计 {len(all_dataset)} 条测试集数据...")
            
            # 及时保存，防止中途断开
            with open(TEST_DATASET_PATH, "w", encoding="utf-8") as f:
                json.dump(all_dataset, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"处理文档 {doc_id} 时出错: {e}")
            continue

    logger.info(f"\n🎉 评测集生成完毕！共 {len(all_dataset)} 条数据，已保存至 {TEST_DATASET_PATH}")
    
    # 统计分类
    type_counts = {}
    for item in all_dataset:
        t = item.get("type", "未知")
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info(f"分类统计: {type_counts}")


if __name__ == "__main__":
    main()
