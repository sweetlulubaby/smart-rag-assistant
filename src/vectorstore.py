"""
src/vectorstore.py - 第二步：向量化与本地存储
使用 HuggingFace 免费开源 Embedding 模型，将文本块转化为向量，
并存储到本地 ChromaDB 向量数据库。
"""
import logging
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    get_embedding_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    从文档块列表构建（或更新）本地 ChromaDB 向量数据库。
    首次运行会下载 Embedding 模型（约 100MB），之后本地缓存。
    
    Args:
        chunks: 切块后的 Document 列表（来自 ingestion.py）
    
    Returns:
        已持久化的 Chroma 向量数据库实例
    """
    logger.info(f"加载 Embedding 模型: 请稍候（首次运行需下载模型）...")
    embeddings = get_embedding_model()
    
    # 确保持久化目录存在
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"正在将 {len(chunks)} 个文本块向量化并存入 ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    
    logger.info(f"✅ 向量库构建完成！数据存储在: {CHROMA_PERSIST_DIR}")
    return vectorstore


def load_vectorstore() -> Optional[Chroma]:
    """
    加载已存在的本地 ChromaDB 向量数据库。
    
    Returns:
        Chroma 实例，如果数据库不存在则返回 None
    """
    persist_path = Path(CHROMA_PERSIST_DIR)
    
    if not persist_path.exists() or not any(persist_path.iterdir()):
        logger.warning(
            f"向量库不存在于 {CHROMA_PERSIST_DIR}，"
            "请先运行 `python scripts/ingest_docs.py` 构建向量库"
        )
        return None
    
    logger.info(f"正在加载已有向量库: {CHROMA_PERSIST_DIR}")
    embeddings = get_embedding_model()
    
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    
    count = vectorstore._collection.count()
    logger.info(f"✅ 向量库加载成功！当前存储 {count} 个向量")
    return vectorstore


def get_vectorstore(chunks: List[Document] = None) -> Chroma:
    """
    智能获取向量数据库：有则加载，传入 chunks 则重建。
    
    Args:
        chunks: 可选，若提供则重新构建向量库
    
    Returns:
        Chroma 实例
    """
    if chunks is not None:
        return build_vectorstore(chunks)
    
    vs = load_vectorstore()
    if vs is None:
        raise RuntimeError(
            "向量库未初始化！请先运行:\n"
            "  python scripts/ingest_docs.py --source ./data/docs/"
        )
    return vs


def delete_vectorstore():
    """清空并删除本地向量数据库（慎用）"""
    import shutil
    persist_path = Path(CHROMA_PERSIST_DIR)
    if persist_path.exists():
        shutil.rmtree(persist_path)
        logger.info(f"已删除向量库: {CHROMA_PERSIST_DIR}")
    else:
        logger.info("向量库不存在，无需删除")
