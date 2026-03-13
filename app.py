"""
app.py - Streamlit 前端界面
智能进阶文档问答助手的 Web UI，支持：
  - PDF 文件上传与实时入库
  - 流式对话（逐 Token 输出，体验如 ChatGPT）
  - 引用来源展示
  - 侧边栏配置（LLM 选择、检索模式等）

启动方式：
  streamlit run app.py
"""
import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import List

import streamlit as st

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

# ─── 页面配置（必须是第一个 Streamlit 调用）─────────────────────────────────
st.set_page_config(
    page_title="智能文档问答助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 自定义 CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 整体背景 */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8e8f0;
}

/* 聊天消息容器 */
.chat-message {
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    margin: 0.8rem 0;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    animation: fadeInUp 0.3s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-message.user {
    background: rgba(88, 101, 242, 0.25);
    border: 1px solid rgba(88, 101, 242, 0.4);
    margin-left: 8%;
}
.chat-message.assistant {
    background: rgba(37, 38, 60, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-right: 8%;
}
.chat-avatar {
    font-size: 1.8rem;
    flex-shrink: 0;
}
.chat-content {
    flex: 1;
    line-height: 1.7;
}

/* 来源引用区域 */
.source-citation {
    background: rgba(255, 255, 255, 0.05);
    border-left: 3px solid #5865f2;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin-top: 0.8rem;
    font-size: 0.82rem;
    color: #a0a0c0;
}

/* 侧边栏 */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.9) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* 标题 */
h1 { 
    background: linear-gradient(120deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* 上传区域 */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(167, 139, 250, 0.4);
    border-radius: 12px;
    padding: 1rem;
}

/* 输入框 */
.stTextInput input, .stChatInput textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e8e8f0 !important;
    border-radius: 12px !important;
}

/* 按钮 */
.stButton > button {
    background: linear-gradient(135deg, #5865f2, #a78bfa);
    border: none;
    color: white;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(88,101,242,0.4);
}

/* 状态指示 */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.status-dot.green { background: #34d399; }
.status-dot.red   { background: #f87171; }
</style>
""", unsafe_allow_html=True)


# ─── 状态初始化 ───────────────────────────────────────────────────────────────
def init_session_state():
    """初始化 Streamlit Session State"""
    defaults = {
        "messages": [],           # 对话历史
        "vectorstore": None,      # 已加载的向量数据库
        "rag_chain": None,        # RAG 链
        "retriever": None,        # 检索器
        "llm": None,              # LLM 实例
        "docs_ingested": False,   # 是否已有文档入库
        "use_multi_query": True,  # 是否使用多查询扩展
        "is_processing": False,   # 是否正在处理
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ─── 组件加载函数（带缓存）────────────────────────────────────────────────────
@st.cache_resource(show_spinner="正在加载向量数据库...")
def load_existing_vectorstore():
    """尝试加载已存在的向量数据库"""
    try:
        from src.vectorstore import load_vectorstore
        vs = load_vectorstore()
        return vs
    except Exception:
        return None


def get_rag_components(use_multi_query: bool = True):
    """初始化或重用 RAG 组件"""
    if st.session_state.rag_chain is None or \
       st.session_state.use_multi_query != use_multi_query:
        
        st.session_state.use_multi_query = use_multi_query
        
        vs = st.session_state.vectorstore
        if vs is None:
            vs = load_existing_vectorstore()
            st.session_state.vectorstore = vs
        
        if vs is None:
            return False
        
        from config import get_llm
        from src.retriever import get_retriever, get_simple_retriever
        from src.rag_chain import get_rag_chain
        
        llm_stream = get_llm(streaming=True)
        llm_query  = get_llm(streaming=False)
        
        if use_multi_query:
            retriever = get_retriever(vs, llm_query)
        else:
            retriever = get_simple_retriever(vs)
        
        st.session_state.retriever = retriever
        st.session_state.rag_chain = get_rag_chain(retriever, llm_stream)
        st.session_state.llm = llm_stream
    
    return st.session_state.rag_chain is not None


# ─── 侧边栏 ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📚 文档问答助手")
    st.markdown("---")
    
    # 向量库状态
    existing_vs = load_existing_vectorstore()
    has_vs = existing_vs is not None or st.session_state.vectorstore is not None
    
    if has_vs:
        st.markdown(
            '<span class="status-dot green"></span>**向量库已就绪**',
            unsafe_allow_html=True,
        )
        if existing_vs and st.session_state.vectorstore is None:
            st.session_state.vectorstore = existing_vs
            st.session_state.docs_ingested = True
    else:
        st.markdown(
            '<span class="status-dot red"></span>**向量库未初始化**',
            unsafe_allow_html=True,
        )
    
    st.markdown("---")
    
    # 文件上传
    st.markdown("### 📂 上传 PDF 文档")
    uploaded_files = st.file_uploader(
        "拖拽或点击上传",
        type=["pdf"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )
    
    if uploaded_files:
        if st.button("📥 开始入库", use_container_width=True):
            with st.spinner("正在处理文档，请稍候..."):
                try:
                    from src.ingestion import chunk_documents, load_pdf
                    from src.vectorstore import build_vectorstore
                    
                    all_chunks = []
                    progress = st.progress(0)
                    for i, uploaded_file in enumerate(uploaded_files):
                        # 保存到临时文件
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        
                        docs = load_pdf(tmp_path)
                        # 修正 metadata 中的来源文件名
                        for doc in docs:
                            doc.metadata["source"] = uploaded_file.name
                        chunks = chunk_documents(docs)
                        all_chunks.extend(chunks)
                        os.unlink(tmp_path)
                        progress.progress((i + 1) / len(uploaded_files))
                    
                    vs = build_vectorstore(all_chunks)
                    st.session_state.vectorstore = vs
                    st.session_state.docs_ingested = True
                    st.session_state.rag_chain = None  # 重置链
                    st.success(f"✅ 成功入库 {len(all_chunks)} 个文本块！")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"入库失败: {str(e)}")
    
    st.markdown("---")
    
    # 检索配置
    st.markdown("### ⚙️ 检索配置")
    use_multi_query = st.toggle(
        "多查询扩展 (Multi-Query)",
        value=True,
        help="使用 LLM 将问题改写为多个角度，提升召回率",
    )
    
    st.markdown("---")
    
    # 清空对话
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown(
        "<small>🔬 基于 LangChain LCEL · ChromaDB · HuggingFace<br>"
        "📊 评估: `python src/evaluation.py`</small>",
        unsafe_allow_html=True,
    )


# ─── 主界面 ───────────────────────────────────────────────────────────────────
st.markdown("# 🤖 智能进阶文档问答助手")
st.markdown(
    "上传你的 PDF 文档，然后用自然语言提问。系统将通过**多查询扩展**精准检索相关段落，"
    "并以**流式输出**实时生成带引用来源的回答。"
)

# 显示历史消息
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    sources = message.get("sources", [])
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="chat-avatar">🧑</div>
            <div class="chat-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = ""
        if sources:
            source_list = "".join(
                f"<div>📄 {s.get('source', '未知')} "
                f"第 {int(s.get('page', 0)) + 1} 页</div>"
                for s in sources[:4]
            )
            sources_html = f'<div class="source-citation">📌 引用来源：{source_list}</div>'
        
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="chat-avatar">🤖</div>
            <div class="chat-content">{content}{sources_html}</div>
        </div>
        """, unsafe_allow_html=True)


# 输入框
question = st.chat_input(
    "请输入你的问题...",
    disabled=not has_vs,
)

if not has_vs:
    st.info("👈 请先在左侧上传 PDF 文档并点击「开始入库」，或确保已运行 `python scripts/ingest_docs.py`")

# 处理用户问题
if question and has_vs:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f"""
    <div class="chat-message user">
        <div class="chat-avatar">🧑</div>
        <div class="chat-content">{question}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 初始化 RAG 组件
    ready = get_rag_components(use_multi_query=use_multi_query)
    
    if not ready:
        st.error("RAG 组件初始化失败，请检查配置和 API Key")
    else:
        chain = st.session_state.rag_chain
        retriever = st.session_state.retriever
        
        # 流式生成答案
        with st.container():
            st.markdown("""
            <div class="chat-message assistant">
                <div class="chat-avatar">🤖</div>
                <div class="chat-content">
            """, unsafe_allow_html=True)
            
            answer_placeholder = st.empty()
            full_answer = ""
            
            try:
                with st.spinner(""):
                    for chunk in chain.stream(question):
                        full_answer += chunk
                        answer_placeholder.markdown(full_answer + "▌")
                    
                    answer_placeholder.markdown(full_answer)
                
                # 获取来源文档
                retrieved_docs = retriever.invoke(question)
                sources = [
                    {
                        "source": Path(doc.metadata.get("source", "未知")).name,
                        "page": doc.metadata.get("page", 0),
                    }
                    for doc in retrieved_docs[:4]
                ]
                
                if sources:
                    source_items = "".join(
                        f"<div>📄 {s['source']} 第 {int(s['page']) + 1} 页</div>"
                        for s in sources
                    )
                    st.markdown(
                        f'<div class="source-citation">📌 引用来源：{source_items}</div>',
                        unsafe_allow_html=True,
                    )
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # 保存到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "sources": sources,
                })
                
            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")
                if "api_key" in str(e).lower():
                    st.warning(
                        "请检查 `.env` 文件中的 API Key 配置是否正确\n\n"
                        "参考 `.env.example` 文件进行配置"
                    )
