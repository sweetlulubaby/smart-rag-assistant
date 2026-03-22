"""
src/vectorstore.py - 第二步：向量化与 FAISS 索引构建
使用 FlagEmbedding (bge-large-zh-v1.5) 做 embedding，
使用 FAISS 构建本地向量索引（默认 Flat，可切换 IVF/HNSW）。

输出：
  - FAISS 索引文件（index.faiss + index.pkl）
  - 索引元信息 JSON（向量维度、文档数、创建时间）
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_DIR,
    FAISS_INDEX_TYPE,
    get_embedding_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _save_index_metadata(
    index_dir: str,
    num_documents: int,
    vector_dim: int,
    index_type: str,
):
    """
    保存索引元信息到 JSON 文件，便于后续查阅和对比。

    Args:
        index_dir: 索引目录
        num_documents: 文档（chunk）数量
        vector_dim: 向量维度
        index_type: 索引类型（Flat/IVF/HNSW）
    """
    metadata = {
        "index_type": index_type,
        "vector_dimension": vector_dim,
        "num_documents": num_documents,
        "embedding_model": "BAAI/bge-large-zh-v1.5",
        "created_at": datetime.now().isoformat(),
    }

    meta_path = Path(index_dir) / "index_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"  索引元信息已保存: {meta_path}")
    logger.info(f"  向量维度: {vector_dim}, 文档数: {num_documents}, 索引类型: {index_type}")


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """
    从文档块列表构建 FAISS 向量索引。
    首次运行会下载 bge-large-zh-v1.5 模型（约 1.3GB），之后本地缓存。

    Args:
        chunks: 切块后的 Document 列表（来自 ingestion.py）

    Returns:
        FAISS 向量数据库实例
    """
    logger.info("加载 Embedding 模型 (bge-large-zh-v1.5): 请稍候（首次运行需下载模型）...")
    embeddings = get_embedding_model()

    # 确保索引目录存在
    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)

    logger.info(f"正在将 {len(chunks)} 个文本块向量化并构建 FAISS {FAISS_INDEX_TYPE} 索引...")

    # 提取文本和元数据（避免 Document.id 兼容性问题）
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )

    # 持久化索引到本地
    vectorstore.save_local(FAISS_INDEX_DIR)
    logger.info(f"✅ FAISS 索引构建完成！数据存储在: {FAISS_INDEX_DIR}")

    # 获取向量维度并保存元信息
    vector_dim = vectorstore.index.d  # FAISS index 的维度属性
    _save_index_metadata(
        index_dir=FAISS_INDEX_DIR,
        num_documents=len(chunks),
        vector_dim=vector_dim,
        index_type=FAISS_INDEX_TYPE,
    )

    return vectorstore


def load_vectorstore() -> Optional[FAISS]:
    """
    加载已存在的 FAISS 向量索引。

    Returns:
        FAISS 实例，如果索引不存在则返回 None
    """
    index_path = Path(FAISS_INDEX_DIR)
    index_file = index_path / "index.faiss"

    if not index_file.exists():
        logger.warning(
            f"FAISS 索引不存在于 {FAISS_INDEX_DIR}，"
            "请先运行 `python scripts/ingest_docs.py` 构建索引"
        )
        return None

    logger.info(f"正在加载 FAISS 索引: {FAISS_INDEX_DIR}")
    embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    num_vectors = vectorstore.index.ntotal
    vector_dim = vectorstore.index.d
    logger.info(
        f"✅ FAISS 索引加载成功！"
        f"当前存储 {num_vectors} 个向量 (维度: {vector_dim})"
    )
    return vectorstore


def get_vectorstore(chunks: List[Document] = None) -> FAISS:
    """
    智能获取向量数据库：有则加载，传入 chunks 则重建。

    Args:
        chunks: 可选，若提供则重新构建索引

    Returns:
        FAISS 实例
    """
    if chunks is not None:
        return build_vectorstore(chunks)

    vs = load_vectorstore()
    if vs is None:
        raise RuntimeError(
            "向量索引未初始化！请先运行：\n"
            "  python scripts/ingest_docs.py --source ./data/docs/"
        )
    return vs


def delete_vectorstore():
    """清空并删除本地 FAISS 索引（慎用）"""
    import shutil
    index_path = Path(FAISS_INDEX_DIR)
    if index_path.exists():
        shutil.rmtree(index_path)
        logger.info(f"已删除 FAISS 索引: {FAISS_INDEX_DIR}")
    else:
        logger.info("FAISS 索引不存在，无需删除")
