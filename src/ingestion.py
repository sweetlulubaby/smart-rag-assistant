"""
src/ingestion.py - 第一步：文档加载与文本切块
支持 PDF 文件（高级解析：布局分析 + 表格提取）和网页 URL 的加载。
使用 RecursiveCharacterTextSplitter 分块，并对表格内容做特殊保护。
"""
import os
import re
import logging
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── PDF 加载（使用新的高级解析器）─────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    """
    加载单个 PDF 文件（高级模式）。
    使用 PyMuPDF + pdfplumber 实现：
    - 多栏布局检测与正确排序
    - 页眉/页脚自动过滤
    - 表格提取为 Markdown 格式

    Args:
        file_path: PDF 文件路径

    Returns:
        Document 对象列表，每个对象对应 PDF 的一页
    """
    from src.pdf_parser import parse_pdf
    return parse_pdf(file_path)


def load_pdfs_from_dir(dir_path: str) -> List[Document]:
    """
    从目录中批量加载所有 PDF 文件（高级解析模式）。

    Args:
        dir_path: 目录路径

    Returns:
        所有 PDF 页面的 Document 对象列表
    """
    from src.pdf_parser import parse_pdfs_from_dir
    return parse_pdfs_from_dir(dir_path)


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


# ─── 表格感知的文本切块 ─────────────────────────────────────────────────────────

def _protect_tables(text: str) -> tuple:
    """
    在切块前将表格替换为占位符，防止表格被切断。

    Args:
        text: 含 [表格 N] ... [/表格 N] 标记的文本

    Returns:
        (替换后文本, {占位符: 原始表格Markdown} 映射)
    """
    table_map = {}
    pattern = re.compile(r'\[表格 (\d+)\]\n(.*?)\n\[/表格 \1\]', re.DOTALL)

    def replacer(match):
        table_id = match.group(1)
        placeholder = f"<<TABLE_PLACEHOLDER_{table_id}>>"
        table_map[placeholder] = match.group(0)
        return placeholder

    protected_text = pattern.sub(replacer, text)
    return protected_text, table_map


def _restore_tables(chunks: List[Document], table_map: dict) -> List[Document]:
    """
    将切块后的文档中的占位符还原为原始表格 Markdown。
    如果一个占位符被完整保留在某个 chunk 中，则还原。

    Args:
        chunks: 切块后的 Document 列表
        table_map: 占位符到原始内容的映射

    Returns:
        还原后的 Document 列表
    """
    restored = []
    for chunk in chunks:
        content = chunk.page_content
        for placeholder, original in table_map.items():
            if placeholder in content:
                content = content.replace(placeholder, original)
        restored.append(Document(
            page_content=content,
            metadata=chunk.metadata,
        ))
    return restored


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    将文档列表切分为更小的块。
    使用 RecursiveCharacterTextSplitter，优先在段落/句子边界处切分。
    对含表格标记的文档，先保护表格不被切断，切块后再还原。

    Args:
        documents: 原始 Document 列表
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块间重叠字符数（保证上下文连贯）

    Returns:
        切块后的 Document 列表
    """
    # 分离含表格和不含表格的文档
    docs_with_tables = []
    docs_without_tables = []

    for doc in documents:
        if doc.metadata.get("has_tables", False) or "[表格 " in doc.page_content:
            docs_with_tables.append(doc)
        else:
            docs_without_tables.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        length_function=len,
    )

    all_chunks = []

    # 1. 处理不含表格的文档 — 直接切块
    if docs_without_tables:
        normal_chunks = splitter.split_documents(docs_without_tables)
        all_chunks.extend(normal_chunks)

    # 2. 处理含表格的文档 — 保护表格后切块
    for doc in docs_with_tables:
        protected_text, table_map = _protect_tables(doc.page_content)

        # 如果保护后文本短于 chunk_size，整体保留
        if len(protected_text) <= chunk_size:
            content = doc.page_content  # 直接用原文（含表格标记）
            all_chunks.append(Document(
                page_content=content,
                metadata=doc.metadata,
            ))
        else:
            # 对保护后的文本切块
            temp_doc = Document(
                page_content=protected_text,
                metadata=doc.metadata,
            )
            temp_chunks = splitter.split_documents([temp_doc])
            # 还原表格
            restored = _restore_tables(temp_chunks, table_map)
            all_chunks.extend(restored)

    logger.info(
        f"文档切块完成: {len(documents)} 原始页 → {len(all_chunks)} 个块 "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    if docs_with_tables:
        logger.info(f"  其中 {len(docs_with_tables)} 页含表格（已保护表格完整性）")

    return all_chunks


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
