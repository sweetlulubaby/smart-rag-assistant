# 智能进阶文档问答助手

> 一个基于 LangChain LCEL 的企业级 RAG 系统，支持 PDF 文档问答、多查询扩展检索、流式输出和自动化 Ragas 评估。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-LCEL-green) ![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI (app.py)                │
│              PDF 上传 │ 流式对话 │ 来源引用展示              │
└───────────────────────────┬─────────────────────────────────┘
                            │ 用户问题
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LCEL RAG Chain (src/rag_chain.py)              │
│                                                             │
│  问题 → MultiQueryRetriever → 向量检索 → Prompt → LLM      │
│              ↓ (多角度子查询)        ↓                      │
│           [子查询1,2,3]        ChromaDB 向量库               │
└───────────────────────────┬─────────────────────────────────┘
                            │ 流式 Token
                            ▼
                      答案 + 引用来源

监控: LangSmith  |  评估: Ragas (Faithfulness, Answer Relevancy)
```

---

## 亮点功能

| 功能 | 说明 |
|------|------|
| **Multi-Query 扩展** | LLM 将问题改写为多个角度，大幅提升召回率 |
| **LCEL 流式输出** | Token 级逐字输出，体验如 ChatGPT |
| **本地 Embedding** | `BAAI/bge-small-zh` 免费开源，无需 API Key |
| **LangSmith 监控** | 每步检索、Prompt、Token 消耗可视化追踪 |
| **Ragas 自动评估** | 量化系统精度：忠实度 + 答案相关性 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

**最低配置**（只需填入 `OPENAI_API_KEY`）：
```env
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
```

**使用本地 Ollama**（无需 API Key，需先安装 [Ollama](https://ollama.ai)）：
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL_NAME=llama3
```

### 3. 准备文档

将你的 PDF 文件放入 `data/docs/` 目录，然后运行：

```bash
python scripts/ingest_docs.py --source ./data/docs/
```

### 4. 启动 Web 界面

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用。

---

## 项目结构

```
smart-rag-assistant/
├── app.py                      # Streamlit 前端界面
├── config.py                   # 配置中心（读取 .env）
├── requirements.txt
├── .env.example                # 环境变量模板
│
├── src/
│   ├── ingestion.py            # Step 1: 文档加载与文本切块
│   ├── vectorstore.py          # Step 2: 向量化与 ChromaDB 存储
│   ├── retriever.py            # Step 3: MultiQueryRetriever 进阶检索
│   ├── rag_chain.py            # Step 4: LCEL 生成管道（流式输出）
│   └── evaluation.py           # Step 5: Ragas 自动化评估
│
├── scripts/
│   └── ingest_docs.py          # CLI 文档入库工具
│
└── data/
    ├── docs/                   # 放置你的 PDF 文件
    ├── chroma_db/              # ChromaDB 向量库（自动生成）
    ├── test_questions.json     # Ragas 评估测试集
    └── evaluation_results.csv  # 评估结果（运行后生成）
```

---

## 使用指南

### 文档入库（CLI）

```bash
# 加载目录中的所有 PDF
python scripts/ingest_docs.py --source ./data/docs/

# 加载指定 PDF 文件
python scripts/ingest_docs.py --files report.pdf thesis.pdf

# 加载网页内容
python scripts/ingest_docs.py --urls https://example.com/article

# 重置向量库并重新入库
python scripts/ingest_docs.py --source ./data/docs/ --reset
```

### Ragas 自动化评估

1. 编辑 `data/test_questions.json`，填入与你文档相关的问题和标准答案
2. 运行评估：

```bash
python src/evaluation.py
```

3. 查看结果：`data/evaluation_results.csv`

评估指标说明：
- **Faithfulness（忠实度）**：答案是否完全基于检索到的文档（0~1，越高越好）
- **Answer Relevancy（答案相关性）**：答案是否真正回答了用户的问题（0~1，越高越好）

---

## LangSmith 链路监控（可选）

1. 注册 [LangSmith](https://smith.langchain.com/) 账号（免费）
2. 在 `.env` 中填入：
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=ls__...
   LANGCHAIN_PROJECT=smart-rag-assistant
   ```
3. 运行任何问答，即可在 LangSmith 面板中看到完整的追踪记录

---

## 核心技术栈

| 组件 | 技术 |
|------|------|
| 核心框架 | LangChain (LCEL) |
| LLM | OpenAI GPT-4o-mini / Ollama Llama3 |
| Embedding | HuggingFace `BAAI/bge-small-zh` |
| 向量数据库 | ChromaDB（本地持久化） |
| 检索策略 | MultiQueryRetriever（多查询扩展） |
| 输出方式 | LCEL Streaming（Token 级流式） |
| 监控 | LangSmith |
| 评估 | Ragas（Faithfulness + Answer Relevancy） |
| 前端 | Streamlit |
