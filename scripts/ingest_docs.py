"""
scripts/ingest_docs.py - 文档入库 CLI 工具
将 PDF 文档或网页批量加载、切块并存入本地 FAISS 向量数据库。

使用示例：
  # 加载目录中的所有 PDF
  python scripts/ingest_docs.py --source ./data/docs/

  # 加载单个 PDF
  python scripts/ingest_docs.py --files ./data/docs/report.pdf

  # 加载网页
  python scripts/ingest_docs.py --urls https://example.com/article

  # 组合使用
  python scripts/ingest_docs.py --source ./data/docs/ --urls https://example.com
"""
import argparse
import logging
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import ingest_sources
from src.vectorstore import build_vectorstore, delete_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="智能文档问答助手 - 文档入库工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        help="包含 PDF 文件的目录路径（递归搜索）",
        default=None,
    )
    parser.add_argument(
        "--files", "-f",
        type=str,
        nargs="+",
        help="单独的 PDF 文件路径列表",
        default=None,
    )
    parser.add_argument(
        "--urls", "-u",
        type=str,
        nargs="+",
        help="网页 URL 列表",
        default=None,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置（清空）向量数据库后重新入库",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查是否有有效的输入源
    if not args.source and not args.files and not args.urls:
        logger.error(
            "请至少指定一个数据源！\n"
            "使用 --source 指定目录，--files 指定文件，或 --urls 指定网页\n"
            "示例: python scripts/ingest_docs.py --source ./data/docs/"
        )
        sys.exit(1)
    
    # 如果需要重置
    if args.reset:
        logger.warning("即将删除现有向量数据库并重新构建...")
        delete_vectorstore()
    
    print("\n" + "="*60)
    print("  智能文档问答助手 - 文档入库")
    print("="*60 + "\n")
    
    # 第一步：加载并切块
    logger.info("步骤 1/2: 加载并切块文档...")
    chunks = ingest_sources(
        pdf_dir=args.source,
        pdf_files=args.files,
        urls=args.urls,
    )
    
    # 第二步：向量化并入库
    logger.info(f"\n步骤 2/2: 向量化 {len(chunks)} 个文本块并存入 FAISS...")
    vectorstore = build_vectorstore(chunks)

    print("\n" + "="*60)
    print(f"  入库完成！共处理 {len(chunks)} 个文本块")
    print("  现在可以运行: streamlit run app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
