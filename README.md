# 智能进阶文档问答助手

一个面向中文金融文档场景的 RAG Demo，支持：
- 高级 PDF 解析：多栏识别、页眉页脚过滤、表格抽取与表格保护切块
- 本地向量检索：`FAISS + bge-large-zh-v1.5`
- 多路召回：`FAISS / BM25 / Hybrid`
- 重排优化：`bge-reranker-v2-m3`
- 保守生成：带证据编号回答、低置信度拒答、回答后校验
- Web Demo：`Streamlit`

当前项目已经完成多轮升级，适合拿来做：
- PDF 文档问答 Demo
- RAG 检索策略实验
- 金融研报 / 财报类问答原型
- 面试或作品集展示

## 1. 当前能力

### 检索链路

默认链路为：

`Hybrid Retrieval -> Top20 Candidate Fetch -> Rerank -> Top5 -> Grounded Answer`

其中包括：
- `FAISS` 稠密召回
- `BM25` 稀疏召回
- 分数加权融合
- `bge-reranker-v2-m3` 重排
- 空召回时 `query rewrite`
- 热点问题缓存

### 生成链路

回答阶段不是直接“让模型自由回答”，而是做了约束：
- 每个关键结论尽量附 `[1]`、`[2]` 证据编号
- 无证据、低置信度、引用异常时自动收敛为“我不确定”
- 回答后会做一次轻量后校验，尽量避免幻觉式数字和无来源结论

### 已完成实验

项目内已经包含以下实验脚本与实验日志：
- 切块策略实验：`scripts/experiment_chunking.py`
- 召回策略实验：`scripts/experiment_retrieval.py`
- 重排实验脚本：`scripts/experiment_rerank.py`
- 升级日志：`upgrade_log.md`

## 2. 项目结构

```text
smart-rag-assistant/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .env.example
├── upgrade_log.md
│
├── src/
│   ├── pdf_parser.py
│   ├── ingestion.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── robust_retriever.py
│   ├── answering.py
│   ├── rag_chain.py
│   ├── langgraph_agent.py
│   └── evaluation.py
│
├── scripts/
│   ├── ingest_docs.py
│   ├── run_demo.ps1
│   ├── experiment_chunking.py
│   ├── experiment_retrieval.py
│   ├── experiment_rerank.py
│   └── generate_eval_dataset.py
│
└── data/
    └── test_questions.json
```

说明：
- `data/docs/` 和 `data/faiss_index/` 属于本地数据，不建议直接提交到 GitHub
- `.env` 含 API Key，不应上传

## 3. 环境要求

- Python `3.10+`
- Windows / macOS / Linux 均可
- 推荐使用虚拟环境
- 首次运行本地 embedding / reranker 时，需要下载 HuggingFace 模型

## 4. 安装

```bash
pip install -r requirements.txt
```

如果你使用仓库内虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 5. 配置

复制环境变量模板：

```bash
cp .env.example .env
```

最低需要配置：

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4o-mini
```

推荐保留当前检索配置：

```env
EMBEDDING_MODEL_NAME=BAAI/bge-large-zh-v1.5
RETRIEVAL_MODE=hybrid
HYBRID_ALPHA=0.75
RERANK_ENABLED=true
RERANK_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANK_FETCH_K=20
RERANK_TOP_N=5
QUERY_REWRITE_ENABLED=true
HOT_QUERY_CACHE_ENABLED=true
```

如果只想跑一个更轻的 Demo，可以这样调低成本：

```env
RERANK_FETCH_K=10
RERANK_TOP_N=5
```

## 6. 准备数据

把你的 PDF 放到：

```text
data/docs/
```

然后执行入库：

```bash
python scripts/ingest_docs.py --source ./data/docs/ --reset
```

这一步会：
- 解析 PDF
- 做表格保护切块
- 用 `bge-large-zh-v1.5` 向量化
- 生成 `FAISS` 本地索引

## 7. 启动 Demo

### 方式 A：直接启动

```bash
streamlit run app.py
```

### 方式 B：Windows 一键启动

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
```

默认地址：

```text
http://localhost:8501
```

说明：
- 第一次提问时可能会慢一些，因为会加载 embedding / reranker
- 如果你开启了 `Multi-Query + Rerank`，CPU 下延迟会明显增加

## 8. 如何做一个顺滑 Demo

如果你只是想现场演示，而不是追求最强效果，建议：

1. 先提前跑过一次，完成模型预热
2. 用比较具体的问题，不要一上来问非常抽象的战略判断题
3. 如果回答偏保守，换成“请引用报告中的依据”这种问法
4. 如果机器较慢，可以先关闭页面里的 `Multi-Query`

推荐提问方式：
- “报告中如何评价 TikTok 电商的发展路径？请引用依据”
- “文档里对公司 2025 年增长的核心判断是什么？请标注来源”
- “列出报告对某行业的 3 个结论，并给出页码”

## 9. 评测与实验

### 切块策略实验

```bash
python scripts/experiment_chunking.py
```

### 召回策略实验

```bash
python scripts/experiment_retrieval.py
```

### 重排实验

```bash
python scripts/experiment_rerank.py
```

### 自动评估

```bash
python src/evaluation.py
```

测试集默认使用：

```text
data/test_questions.json
```

## 10. 当前技术栈

- `LangChain`
- `LangGraph`
- `Streamlit`
- `FAISS`
- `BM25`
- `FlagEmbedding`
- `PyMuPDF`
- `pdfplumber`
- `Ragas`

## 11. 注意事项

- `.env` 不要上传到 GitHub
- `data/docs/` 不要上传真实业务 PDF
- `data/faiss_index/` 属于可重建索引，通常不建议入库
- 如果启用了 `LangSmith`，请确认你的网络与 API Key 都正常
- `bge-reranker-v2-m3` 在 CPU 上较慢，完整实验可能需要很久

## 12. 后续可继续做的方向

- 给 reranker 增加 GPU 推理支持
- 做更严格的 citation-to-source 对齐
- 增加多文档对比问答模板
- 增加面向 Demo 的“快速模式 / 精准模式”切换
