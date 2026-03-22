"""
src/rag_chain.py - 第四步：使用 LCEL 组装生成管道
通过 LangChain Expression Language (LCEL) 将检索器、Prompt、LLM 串联成链，
支持 Token 级流式输出（Streaming），响应时像 ChatGPT 一样逐字吐出。
"""
import logging
from pathlib import Path
from typing import AsyncIterator, Iterator, List, Dict, Any

from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from langchain.schema import BaseRetriever

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.answering import generate_grounded_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Prompt 模板 ──────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """你是一个专业的文档问答助手。请严格根据以下参考文档来回答用户的问题。

要求：
1. 只使用参考文档中的信息作答，不要编造内容
2. 如果参考文档中没有相关信息，直接告知用户"根据现有文档，无法找到相关信息"
3. 回答请简洁、准确、有条理

参考文档：
{context}

用户问题：{question}

请根据以上参考文档回答："""


def format_docs(docs: List[Document]) -> str:
    """
    将检索到的文档列表格式化为 Prompt 上下文字符串。
    包含来源信息（文件名 + 页码）供 LLM 引用。
    
    Args:
        docs: 检索到的 Document 列表
    
    Returns:
        格式化的上下文字符串
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知文档")
        # 提取文件名，避免显示完整路径
        source_name = Path(source).name if source != "未知文档" else source
        page = doc.metadata.get("page", "")
        page_info = f" 第{int(page) + 1}页" if page != "" else ""
        
        parts.append(
            f"【参考文档 {i}】({source_name}{page_info})\n{doc.page_content}"
        )
    
    return "\n\n---\n\n".join(parts)


class SimpleRAGChain:
    def __init__(self, retriever: BaseRetriever, llm: BaseLanguageModel):
        self.retriever = retriever
        self.llm = llm

    def stream(self, question: str):
        docs = self.retriever.invoke(question)
        result = generate_grounded_answer(self.llm, question, docs)
        yield str(result.get("answer", ""))

    def invoke(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        result = generate_grounded_answer(self.llm, question, docs)
        return str(result.get("answer", ""))


def get_rag_chain(retriever: BaseRetriever, llm: BaseLanguageModel):
    logger.info("✅ 简易 RAG 链构建完成（兼容当前 LangChain 版本）")
    return SimpleRAGChain(retriever, llm)


def get_rag_chain_with_sources(retriever: BaseRetriever, llm: BaseLanguageModel):
    return get_rag_chain(retriever, llm)


def stream_answer(chain, question: str) -> Iterator[str]:
    yield from chain.stream(question)


async def astream_answer(chain, question: str) -> AsyncIterator[str]:
    """
    异步流式生成回答（适用于 async 框架）。
    
    Args:
        chain: LCEL RAG 链
        question: 用户问题
    
    Yields:
        每个 token 字符串
    """
    async for chunk in chain.astream(question):
        yield chunk
