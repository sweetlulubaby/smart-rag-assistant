"""
src/pdf_parser.py - 高级 PDF 解析模块
参考 RAGFlow DeepDoc 思路，使用：
  - PyMuPDF (fitz)：提取带坐标的文本块，实现布局分析（多栏检测、页眉页脚过滤）
  - pdfplumber：精准提取表格，转为 Markdown 格式保留行列结构

核心解决的问题：
  1. 多栏 PDF 内容按栏顺序正确拼接（而非跨栏行拼接）
  2. 表格提取为 Markdown 表格，保留结构信息
  3. 自动过滤页眉/页脚噪声文本
  4. 表格区域与正文区域分离，避免表格被切块器拆散
"""
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import fitz  # PyMuPDF
import pdfplumber
from langchain.schema import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PDF_HEADER_RATIO, PDF_FOOTER_RATIO, PDF_MIN_TEXT_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── 表格提取（pdfplumber）─────────────────────────────────────────────────────

def extract_tables_from_page(
    plumber_page: pdfplumber.page.Page,
) -> List[Dict[str, Any]]:
    """
    使用 pdfplumber 提取页面中的所有表格。

    Returns:
        列表，每个元素包含:
        - "bbox": (x0, y0, x1, y1) 表格边界框
        - "markdown": 转换后的 Markdown 表格字符串
    """
    tables_info = []

    try:
        tables = plumber_page.find_tables()
    except Exception as e:
        logger.debug(f"pdfplumber 表格检测失败: {e}")
        return tables_info

    for table in tables:
        try:
            bbox = table.bbox  # (x0, y0, x1, y1)
            data = table.extract()

            if not data or len(data) < 2:
                continue

            markdown = _table_to_markdown(data)
            if markdown:
                tables_info.append({
                    "bbox": bbox,
                    "markdown": markdown,
                })
        except Exception as e:
            logger.debug(f"表格提取失败: {e}")
            continue

    return tables_info


def _table_to_markdown(table_data: List[List[Optional[str]]]) -> str:
    """
    将二维表格数据转换为 Markdown 表格格式。

    Args:
        table_data: 二维列表，第一行视为表头

    Returns:
        Markdown 表格字符串
    """
    if not table_data:
        return ""

    # 清理单元格内容
    cleaned = []
    for row in table_data:
        cleaned_row = []
        for cell in row:
            cell_text = str(cell).strip() if cell is not None else ""
            # 将单元格内的换行替换为空格
            cell_text = re.sub(r'\s+', ' ', cell_text)
            cleaned_row.append(cell_text)
        cleaned.append(cleaned_row)

    # 统一列数（取最大列数）
    max_cols = max(len(row) for row in cleaned)
    for row in cleaned:
        while len(row) < max_cols:
            row.append("")

    # 过滤全空行
    cleaned = [row for row in cleaned if any(cell.strip() for cell in row)]
    if not cleaned:
        return ""

    # 构建 Markdown 表格
    lines = []
    # 表头
    header = cleaned[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    # 数据行
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ─── 布局感知文本提取（PyMuPDF）──────────────────────────────────────────────

def _is_in_bbox(block_rect: Tuple[float, ...], bbox: Tuple[float, ...],
                tolerance: float = 5.0) -> bool:
    """
    判断一个文本块是否在某个边界框内（带容差）。

    Args:
        block_rect: 文本块的 (x0, y0, x1, y1)
        bbox: 边界框的 (x0, y0, x1, y1)
        tolerance: 像素容差
    """
    return (block_rect[0] >= bbox[0] - tolerance and
            block_rect[1] >= bbox[1] - tolerance and
            block_rect[2] <= bbox[2] + tolerance and
            block_rect[3] <= bbox[3] + tolerance)


def _detect_columns(blocks: List[Dict], page_width: float) -> List[List[Dict]]:
    """
    基于文本块的 x 坐标分布检测多栏布局。

    思路（参考 RAGFlow DeepDoc）：
    1. 收集所有文本块的 x0 坐标
    2. 如果存在明显的 x 坐标分组（间距 > 页面宽度的 10%），则判定为多栏
    3. 按栏分组，每栏内按 y 坐标排序

    Args:
        blocks: PyMuPDF 文本块列表
        page_width: 页面宽度

    Returns:
        按栏分组后的文本块列表（每组内已按 y 排序）
    """
    if not blocks:
        return []

    # 收集所有 block 的 x0 坐标
    x0_values = sorted(set(round(b["x0"], 0) for b in blocks))

    if len(x0_values) <= 1:
        # 单栏
        return [sorted(blocks, key=lambda b: b["y0"])]

    # 寻找 x0 坐标之间的大间隔来分栏
    gap_threshold = page_width * 0.10
    column_boundaries = [x0_values[0]]

    for i in range(1, len(x0_values)):
        if x0_values[i] - x0_values[i - 1] > gap_threshold:
            column_boundaries.append(x0_values[i])

    if len(column_boundaries) <= 1:
        # 没有明显分栏
        return [sorted(blocks, key=lambda b: b["y0"])]

    # 按栏分组
    columns = [[] for _ in column_boundaries]
    for block in blocks:
        # 找最近的栏边界
        min_dist = float("inf")
        col_idx = 0
        for i, boundary in enumerate(column_boundaries):
            dist = abs(block["x0"] - boundary)
            if dist < min_dist:
                min_dist = dist
                col_idx = i
        columns[col_idx].append(block)

    # 每栏内按 y 排序
    columns = [sorted(col, key=lambda b: b["y0"]) for col in columns if col]

    if len(columns) > 1:
        logger.debug(f"  检测到 {len(columns)} 栏布局")

    return columns


def extract_text_with_layout(
    fitz_page: fitz.Page,
    table_bboxes: List[Tuple[float, ...]] = None,
) -> str:
    """
    使用 PyMuPDF 提取页面文本，带布局感知：
    - 自动检测并处理多栏布局
    - 过滤页眉/页脚区域
    - 排除已被表格占据的区域

    Args:
        fitz_page: PyMuPDF 页面对象
        table_bboxes: 需要排除的表格区域列表

    Returns:
        布局感知的纯文本内容
    """
    if table_bboxes is None:
        table_bboxes = []

    page_rect = fitz_page.rect
    page_height = page_rect.height
    page_width = page_rect.width

    # 定义页眉/页脚区域
    header_limit = page_height * PDF_HEADER_RATIO
    footer_limit = page_height * (1 - PDF_FOOTER_RATIO)

    # 获取文本块信息
    text_dict = fitz_page.get_text("dict")
    blocks_raw = text_dict.get("blocks", [])

    # 过滤文本块
    text_blocks = []
    for block in blocks_raw:
        # 只处理文本块（type=0），跳过图像块（type=1）
        if block.get("type") != 0:
            continue

        x0, y0, x1, y1 = block["bbox"]

        # 过滤页眉区域
        if y1 <= header_limit:
            continue

        # 过滤页脚区域
        if y0 >= footer_limit:
            continue

        # 排除表格区域
        in_table = False
        for t_bbox in table_bboxes:
            if _is_in_bbox((x0, y0, x1, y1), t_bbox, tolerance=10.0):
                in_table = True
                break
        if in_table:
            continue

        # 提取块内文本
        block_text = ""
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            if line_text.strip():
                block_text += line_text.strip() + "\n"

        block_text = block_text.strip()
        if len(block_text) < PDF_MIN_TEXT_LENGTH:
            continue

        text_blocks.append({
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "text": block_text,
        })

    # 多栏检测与排序
    columns = _detect_columns(text_blocks, page_width)

    # 按栏顺序拼接文本
    result_parts = []
    for column in columns:
        for block in column:
            result_parts.append(block["text"])

    return "\n\n".join(result_parts)


# ─── 页面级合并 ─────────────────────────────────────────────────────────────────

def parse_pdf_page(
    fitz_page: fitz.Page,
    plumber_page: pdfplumber.page.Page,
    page_num: int,
) -> str:
    """
    解析单个 PDF 页面：提取表格 + 布局感知文本，智能合并。

    策略：
    1. 先用 pdfplumber 提取表格，获取表格区域坐标
    2. 用 PyMuPDF 提取非表格区域的文本（带多栏/页眉页脚处理）
    3. 将表格 Markdown 按其在页面中的 y 坐标位置插入到合适的文本位置

    Args:
        fitz_page: PyMuPDF 页面对象
        plumber_page: pdfplumber 页面对象
        page_num: 页码（0-indexed）

    Returns:
        合并后的页面文本（正文 + Markdown 表格）
    """
    # 1. 提取表格
    tables_info = extract_tables_from_page(plumber_page)
    table_bboxes = [t["bbox"] for t in tables_info]

    if tables_info:
        logger.debug(f"  第 {page_num + 1} 页: 提取到 {len(tables_info)} 个表格")

    # 2. 提取布局感知文本（排除表格区域）
    text_content = extract_text_with_layout(fitz_page, table_bboxes)

    # 3. 如果有表格，将表格 Markdown 附加到文本末尾
    #    （使用明确的分隔标记，方便切块器识别）
    if tables_info:
        table_sections = []
        for i, table in enumerate(tables_info):
            table_sections.append(
                f"\n\n[表格 {i + 1}]\n{table['markdown']}\n[/表格 {i + 1}]"
            )
        text_content = text_content + "\n" + "\n".join(table_sections)

    return text_content


# ─── 文档级入口 ─────────────────────────────────────────────────────────────────

def parse_pdf(file_path: str) -> List[Document]:
    """
    高级 PDF 解析入口：同时使用 PyMuPDF 和 pdfplumber 解析整个 PDF。

    功能：
    - 多栏布局检测与正确排序
    - 页眉/页脚自动过滤
    - 表格提取为 Markdown 格式
    - 返回 LangChain Document 对象列表

    Args:
        file_path: PDF 文件路径

    Returns:
        Document 对象列表，每页一个 Document
    """
    file_path = str(Path(file_path).resolve())
    logger.info(f"正在解析 PDF（高级模式）: {file_path}")

    documents = []
    total_tables = 0

    # 同时打开两个 PDF 处理器
    fitz_doc = fitz.open(file_path)
    plumber_doc = pdfplumber.open(file_path)

    try:
        num_pages = len(fitz_doc)
        logger.info(f"  共 {num_pages} 页")

        for page_num in range(num_pages):
            fitz_page = fitz_doc[page_num]
            plumber_page = plumber_doc.pages[page_num]

            # 解析页面
            page_text = parse_pdf_page(fitz_page, plumber_page, page_num)

            if not page_text.strip():
                logger.debug(f"  第 {page_num + 1} 页: 无有效内容，跳过")
                continue

            # 统计表格数
            page_tables = page_text.count("[表格 ")
            total_tables += page_tables

            # 构建 Document
            doc = Document(
                page_content=page_text,
                metadata={
                    "source": file_path,
                    "page": page_num,
                    "has_tables": page_tables > 0,
                    "parser": "pymupdf+pdfplumber",
                },
            )
            documents.append(doc)

    finally:
        fitz_doc.close()
        plumber_doc.close()

    logger.info(
        f"  ✅ PDF 解析完成: {len(documents)} 页有效内容, "
        f"共提取 {total_tables} 个表格"
    )
    return documents


def parse_pdfs_from_dir(dir_path: str) -> List[Document]:
    """
    从目录中批量解析所有 PDF 文件。

    Args:
        dir_path: 目录路径

    Returns:
        所有 PDF 页面的 Document 对象列表
    """
    dir_path = Path(dir_path)
    pdf_files = sorted(dir_path.glob("**/*.pdf"))

    if not pdf_files:
        logger.warning(f"目录 {dir_path} 中没有找到 PDF 文件")
        return []

    all_docs = []
    for pdf_file in pdf_files:
        docs = parse_pdf(str(pdf_file))
        all_docs.extend(docs)

    logger.info(f"共解析 {len(pdf_files)} 个 PDF, 总计 {len(all_docs)} 页")
    return all_docs
