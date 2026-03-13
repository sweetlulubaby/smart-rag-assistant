"""
src/rag_chain.py - 第四步：使用 LCEL 组装生成管道
通过 LangChain Expression Language (LCEL) 将检索器、Prompt、LLM 串联成链，
支持 Token 级流式输出（Streaming），响应时像 ChatGPT 一样逐字吐出。
"""
import logging
from pathlib import Path
from typing import AsyncIterator, Iterator, List, Dict, Any

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.language_models import BaseLanguageModel
from langchain.schema import BaseRetriever

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Prompt 模板 ──────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """你是一个专业的文档问答助手。请严格根据以下参考文档来回答用户的问题。

要求：
1. 只使用参考文档中的信息作答，不要编造内容
2. 如果参考文档中没有相关信息，直接告知用户"根据现有文档，无法找到相关信息"
3. 回答时指出信息来自哪个参考文档（使用【参考文档 X】标注）
4. 回答要简洁、准确、有条理

参考文档：
{context}

用户问题：{question}

请根据以上参考文档回答："""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


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


def get_rag_chain(retriever: BaseRetriever, llm: BaseLanguageModel):
    """
    构建 LCEL RAG 链。
    
    数据流（LCEL 管道）：
      用户问题
        ↓
      并行执行：
        - retrieved_docs = retriever.invoke(question)  → 格式化为 context
        - question = question (直通)
        ↓
      RAG_PROMPT.format(context=..., question=...)
        ↓
      llm.invoke(prompt)  [支持 streaming]
        ↓
      StrOutputParser()  → 最终字符串
    
    Args:
        retriever: 检索器（MultiQueryRetriever 或简单检索器）
        llm: LLM 实例（需支持 streaming=True）
    
    Returns:
        LCEL 可执行链
    """
    rag_chain_with_source = RunnableParallel(
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
    )
    
    chain = rag_chain_with_source | RAG_PROMPT | llm | StrOutputParser()
    
    logger.info("✅ LCEL RAG 链构建完成（支持 Streaming）")
    return chain


def get_rag_chain_with_sources(retriever: BaseRetriever, llm: BaseLanguageModel):
    """
    构建带来源文档返回的 RAG 链（适用于前端展示引用来源）。
    
    Returns:
        Dict 格式：{"answer": str, "source_documents": List[Document]}
    """
    # 子链：格式化用于生成答案
    answer_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    
    # 子链：获取原始文档（用于展示来源）
    source_chain = RunnableParallel(
        {
            "answer": answer_chain,
            "source_documents": retriever,
        }
    )
    
    return source_chain


def stream_answer(chain, question: str) -> Iterator[str]:
    """
    流式生成回答（逐 token 输出）。
    
    Args:
        chain: LCEL RAG 链
        question: 用户问题
    
    Yields:
        每个 token 字符串
    """
    for chunk in chain.stream(question):
        yield chunk


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
