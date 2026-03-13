"""
src/retriever.py - 第三步：实现进阶检索策略
使用 MultiQueryRetriever 进行多查询扩展，解决普通单次搜索召回率低的问题。

原理：LLM 将用户问题改写成多个不同角度的子问题，分别检索后合并去重，
      从而大幅提升相关文档的召回率。
"""
import logging
from pathlib import Path
from typing import List

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document
from langchain_core.language_models import BaseLanguageModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RETRIEVER_K

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 开启 MultiQueryRetriever 的内部日志，可以看到生成了哪些子查询
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def get_retriever(vectorstore: Chroma, llm: BaseLanguageModel) -> MultiQueryRetriever:
    """
    构建 MultiQueryRetriever（多查询检索器）。

    工作流程：
    1. 接收用户原始问题（如"怎么避税？"）
    2. 使用 LLM 生成多个角度的子查询（如"合法节税方式"、"税务筹划策略"等）
    3. 用每个子查询分别搜索向量库
    4. 合并所有结果并去重
    5. 返回综合检索结果

    Args:
        vectorstore: 已加载的 Chroma 向量数据库
        llm: LLM 实例（用于生成子查询）

    Returns:
        MultiQueryRetriever 实例
    """
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    logger.info(
        f"✅ MultiQueryRetriever 构建完成 "
        f"(每子查询返回 top-{RETRIEVER_K} 文档)"
    )
    return retriever


def get_simple_retriever(vectorstore: Chroma) -> BaseRetriever:
    """
    构建简单的相似度检索器（不使用多查询扩展）。
    适用于快速测试或 LLM 资源受限场景。

    Args:
        vectorstore: 已加载的 Chroma 向量数据库

    Returns:
        基础向量相似度检索器
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


def format_retrieved_docs(docs: List[Document]) -> str:
    """
    将检索到的文档列表格式化为字符串，供 Prompt 使用。
    每个文档附带来源信息（文件名 + 页码）。

    Args:
        docs: 检索到的 Document 列表

    Returns:
        格式化后的上下文字符串
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
