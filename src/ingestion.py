"""
src/ingestion.py - 第一步：文档加载与文本切块
支持 PDF 文件和网页 URL 的加载，使用 RecursiveCharacterTextSplitter 分块。
"""
import os
import logging
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[Document]:
    """
    加载单个 PDF 文件。
    
    Args:
        file_path: PDF 文件路径
    
    Returns:
        Document 对象列表，每个对象对应 PDF 的一页
    """
    logger.info(f"正在加载 PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    logger.info(f"  已加载 {len(docs)} 页")
    return docs


def load_pdfs_from_dir(dir_path: str) -> List[Document]:
    """
    从目录中批量加载所有 PDF 文件。
    
    Args:
        dir_path: 目录路径
    
    Returns:
        所有 PDF 页面的 Document 对象列表
    """
    dir_path = Path(dir_path)
    pdf_files = list(dir_path.glob("**/*.pdf"))
    
    if not pdf_files:
        logger.warning(f"目录 {dir_path} 中没有找到 PDF 文件")
        return []
    
    all_docs = []
    for pdf_file in pdf_files:
        docs = load_pdf(str(pdf_file))
        all_docs.extend(docs)
    
    logger.info(f"共加载 {len(pdf_files)} 个 PDF, 总计 {len(all_docs)} 页")
    return all_docs


def load_web_pages(urls: List[str]) -> List[Document]:
    """
    加载网页内容。
    
    Args:
        urls: URL 字符串列表
    
    Returns:
        Document 对象列表
    """
    logger.info(f"正在加载 {len(urls)} 个网页...")
    loader = WebBaseLoader(urls)
    docs = loader.load()
    logger.info(f"  已加载 {len(docs)} 个网页")
    return docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    将文档列表切分为更小的块。
    使用 RecursiveCharacterTextSplitter，优先在段落/句子边界处切分，
    避免语义硬截断。
    
    Args:
        documents: 原始 Document 列表
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块间重叠字符数（保证上下文连贯）
    
    Returns:
        切块后的 Document 列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(
        f"文档切块完成: {len(documents)} 原始页 → {len(chunks)} 个块 "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def ingest_sources(
    pdf_dir: str = None,
    pdf_files: List[str] = None,
    urls: List[str] = None,
) -> List[Document]:
    """
    统一入口：从多种来源加载并切块文档。
    
    Args:
        pdf_dir: PDF 目录路径（批量加载）
        pdf_files: 单独 PDF 文件路径列表
        urls: 网页 URL 列表
    
    Returns:
        切块后的 Document 列表，可直接入库向量数据库
    """
    all_docs = []
    
    if pdf_dir and Path(pdf_dir).exists():
        all_docs.extend(load_pdfs_from_dir(pdf_dir))
    
    if pdf_files:
        for f in pdf_files:
            all_docs.extend(load_pdf(f))
    
    if urls:
        all_docs.extend(load_web_pages(urls))
    
    if not all_docs:
        raise ValueError("没有加载到任何文档，请检查输入路径或 URL")
    
    return chunk_documents(all_docs)
