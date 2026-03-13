"""
config.py - 项目配置中心
从 .env 文件加载所有配置项
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ─── LLM 配置 ────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3")

# ─── Embedding 配置 ───────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh")

# ─── ChromaDB 配置 ────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME = "rag_documents"

# ─── 文档切块参数 ─────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ─── 检索参数 ─────────────────────────────────────────────────
RETRIEVER_K = 4               # 每个子查询返回的文档数
MULTI_QUERY_COUNT = 3         # 多查询扩展生成的子查询数量

# ─── 评估配置 ─────────────────────────────────────────────────
TEST_DATASET_PATH = os.getenv("TEST_DATASET_PATH", "./data/test_questions.json")
EVAL_RESULTS_PATH = "./data/evaluation_results.csv"

# ─── LangSmith 配置 (自动被 LangChain 读取) ──────────────────
# LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
# 这些变量由 LangChain 自动从环境变量读取，无需手动配置


def get_llm(streaming: bool = False):
    """
    根据 LLM_PROVIDER 配置返回对应的 LLM 实例。
    
    Args:
        streaming: 是否开启流式输出
    
    Returns:
        LangChain LLM 实例
    """
    if LLM_PROVIDER == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL_NAME,
            streaming=streaming,
        )
    else:
        # 默认使用 OpenAI
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=OPENAI_MODEL_NAME,
            streaming=streaming,
            temperature=0.0,
        )


def get_embedding_model():
    """
    返回 HuggingFace Embedding 模型实例 (本地运行, 无需 API Key)。
    
    Returns:
        HuggingFaceEmbeddings 实例
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
