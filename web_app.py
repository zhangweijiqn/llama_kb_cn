"""
基于 Streamlit 的知识库问答系统 Web 界面
"""
import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from kb_qa import KnowledgeBaseQA

# 页面配置
st.set_page_config(
    page_title="知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'kb_built' not in st.session_state:
    st.session_state.kb_built = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'only_knowledge_base' not in st.session_state:
    st.session_state.only_knowledge_base = True  # 默认仅在知识库内回答
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.1  # 默认不过滤


def init_qa_system():
    """初始化问答系统"""
    if st.session_state.qa_system is None:
        with st.spinner("正在初始化系统..."):
            try:
                st.session_state.qa_system = KnowledgeBaseQA(
                    vector_store_path='web_knowledge_base',
                    llama_model_name="Qwen/Qwen2-0.5B-Instruct",  # 更小的中文模型，约500MB
                    use_quantization=True,  # 使用量化以节省内存
                    only_knowledge_base=st.session_state.only_knowledge_base,
                    similarity_threshold=0.0  # 初始化时不设置阈值，使用页面传入的值
                )
                # 检查是否已有知识库内容
                if st.session_state.qa_system.has_content():
                    st.session_state.kb_built = True
                return True
            except Exception as e:
                st.error(f"初始化失败: {e}")
                return False
    else:
        # 更新 only_knowledge_base 设置（similarity_threshold 在查询时动态传入）
        st.session_state.qa_system.only_knowledge_base = st.session_state.only_knowledge_base
    return True


def check_knowledge_base_exists():
    """检查知识库是否存在"""
    if st.session_state.qa_system:
        return st.session_state.qa_system.has_content()
    return False


def clear_knowledge_base():
    """清空知识库"""
    if st.session_state.qa_system:
        try:
            st.session_state.qa_system.clear_knowledge_base()
            st.session_state.kb_built = False
            st.session_state.chat_history = []
            return True
        except Exception as e:
            st.error(f"清空失败: {e}")
            return False
    return False


def build_knowledge_base_from_files(uploaded_files, temp_dir):
    """从上传的文件构建知识库"""
    if not uploaded_files:
        st.warning("请先上传文档！")
        return False
    
    if not init_qa_system():
        return False
    
    # 保存上传的文件到临时目录
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
    
    # 构建知识库
    with st.spinner("正在构建知识库，这可能需要一些时间..."):
        try:
            if len(saved_files) == 1:
                st.session_state.qa_system.build_knowledge_base(
                    saved_files[0],
                    chunk_size=500,
                    chunk_overlap=50
                )
            else:
                # 多个文件，创建临时目录
                multi_dir = os.path.join(temp_dir, "documents")
                os.makedirs(multi_dir, exist_ok=True)
                for file_path in saved_files:
                    shutil.move(file_path, multi_dir)
                st.session_state.qa_system.build_knowledge_base(
                    multi_dir,
                    chunk_size=500,
                    chunk_overlap=50
                )
            
            st.session_state.kb_built = True
            stats = st.session_state.qa_system.get_stats()
            st.success(f"✓ 知识库构建完成！共 {stats['total_documents']} 个文档片段")
            return True
        except Exception as e:
            st.error(f"构建失败: {e}")
            return False


def build_knowledge_base_from_path(document_path):
    """从文件路径构建知识库"""
    if not document_path or not os.path.exists(document_path):
        st.warning("请提供有效的文档路径！")
        return False
    
    if not init_qa_system():
        return False
    
    # 构建知识库
    with st.spinner("正在构建知识库，这可能需要一些时间..."):
        try:
            st.session_state.qa_system.build_knowledge_base(
                document_path,
                chunk_size=500,
                chunk_overlap=50
            )
            
            st.session_state.kb_built = True
            stats = st.session_state.qa_system.get_stats()
            st.success(f"✓ 知识库构建完成！共 {stats['total_documents']} 个文档片段")
            return True
        except Exception as e:
            st.error(f"构建失败: {e}")
            return False


def query_knowledge_base(question):
    """查询知识库"""
    if not st.session_state.kb_built:
        st.warning("请先构建知识库！")
        return None
    
    if not question.strip():
        return None
    
    with st.spinner("正在检索和生成答案..."):
        try:
            result = st.session_state.qa_system.query(
                question,
                top_k=3,
                max_new_tokens=256,  # 使用 max_new_tokens 避免输入长度问题
                temperature=0.7,
                similarity_threshold=st.session_state.similarity_threshold
            )
            return result
        except Exception as e:
            st.error(f"查询失败: {e}")
            return None


# 主界面
st.markdown('<div class="main-header">📚 知识库问答系统</div>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    
    # 初始化系统
    if st.button("🔄 初始化系统"):
        st.session_state.qa_system = None
        st.session_state.kb_built = False
        st.session_state.chat_history = []
        if init_qa_system():
            st.success("系统初始化成功！")
    
    st.divider()
    
    # 知识库状态
    st.subheader("📊 知识库状态")
    if st.session_state.qa_system:
        if st.session_state.qa_system.has_content():
            stats = st.session_state.qa_system.get_stats()
            st.success("✓ 知识库已加载")
            st.info(f"""
            **文档片段数**: {stats['total_documents']}  
            **向量维度**: {stats['dimension']}  
            **索引大小**: {stats['index_size']}
            """)
            st.session_state.kb_built = True
        else:
            st.warning("知识库为空")
    else:
        st.warning("系统未初始化")
    
    st.divider()
    
    # 清空知识库（如果有内容）
    if st.session_state.qa_system and st.session_state.qa_system.has_content():
        if st.button("🗑️ 清空知识库", type="secondary"):
            if clear_knowledge_base():
                st.success("知识库已清空")
                st.rerun()
    
    st.divider()
    
    # 清空历史
    if st.button("🗑️ 清空对话历史"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # 回答模式设置
    st.subheader("⚙️ 回答模式")
    only_kb = st.checkbox(
        "仅在知识库内回答",
        value=st.session_state.only_knowledge_base,
        help="开启：仅基于知识库内容回答；关闭：可以回答知识库之外的问题"
    )
    
    if only_kb != st.session_state.only_knowledge_base:
        st.session_state.only_knowledge_base = only_kb
        if st.session_state.qa_system:
            st.session_state.qa_system.only_knowledge_base = only_kb
        st.info("回答模式已更新" if only_kb else "已允许回答知识库外的问题")
    
    st.divider()
    
    # 相似度阈值设置
    st.subheader("🎯 相关性阈值")
    threshold = st.slider(
        "相似度阈值",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.similarity_threshold,
        step=0.05,
        help="过滤低相关性结果。值越高，只返回越相关的结果。\n"
             "• 0.0: 不过滤（返回所有结果）\n"
             "• 0.5: 中等严格（过滤相似度 < 50% 的结果）\n"
             "• 0.7: 较严格（过滤相似度 < 70% 的结果）\n"
             "• 0.8: 很严格（只返回相似度 >= 80% 的结果）"
    )
    
    if threshold != st.session_state.similarity_threshold:
        st.session_state.similarity_threshold = threshold
        if st.session_state.qa_system:
            st.session_state.qa_system.similarity_threshold = threshold
        st.info(f"相似度阈值已更新为 {threshold:.0%}")
    
    # 显示阈值说明
    if threshold == 0.0:
        st.caption("📌 当前：不过滤，返回所有检索结果")
    elif threshold < 0.5:
        st.caption(f"📌 当前：宽松模式（相似度 >= {threshold:.0%}）")
    elif threshold < 0.7:
        st.caption(f"📌 当前：中等严格（相似度 >= {threshold:.0%}）")
    elif threshold < 0.8:
        st.caption(f"📌 当前：较严格（相似度 >= {threshold:.0%}）")
    else:
        st.caption(f"📌 当前：很严格（相似度 >= {threshold:.0%}）")
    
    st.divider()
    
    # 帮助信息
    with st.expander("ℹ️ 使用帮助"):
        st.markdown("""
        **使用步骤：**
        1. 上传文档（支持 PDF、Word、TXT、Markdown）
        2. 点击"构建知识库"
        3. 在输入框中输入问题
        4. 查看答案和来源
        
        **支持的功能：**
        - 多文档上传
        - 语义检索
        - 答案生成
        - 来源追溯
        - 知识库内外回答模式切换
        """)


# 主内容区域
tab1, tab2, tab3 = st.tabs(["📄 文档管理", "💬 问答", "📊 统计"])

# Tab 1: 文档管理
with tab1:
    st.header("📄 文档上传与管理")
    
    # 初始化系统（如果未初始化）
    if not st.session_state.qa_system:
        if st.button("🚀 初始化系统", type="primary"):
            if init_qa_system():
                st.rerun()
    else:
        # 检查是否已有知识库
        if st.session_state.qa_system.has_content():
            st.success("✓ 检测到已存在的知识库")
            stats = st.session_state.qa_system.get_stats()
            st.info(f"当前知识库包含 **{stats['total_documents']}** 个文档片段")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 使用现有知识库", type="primary", use_container_width=True):
                    st.session_state.kb_built = True
                    st.success("已加载现有知识库，可以直接开始问答！")
                    st.rerun()
            with col2:
                if st.button("🗑️ 清空知识库", type="secondary", use_container_width=True):
                    if clear_knowledge_base():
                        st.rerun()
            
            st.divider()
            st.markdown("**或者重新构建知识库：**")
        
        # 文档来源选择
        source_option = st.radio(
            "选择文档来源",
            ["上传文件", "从文件系统选择"],
            horizontal=True
        )
        
        if source_option == "上传文件":
            # 文件上传
            uploaded_files = st.file_uploader(
                "上传文档",
                type=['txt', 'md', 'pdf', 'docx'],
                accept_multiple_files=True,
                help="支持 TXT、Markdown、PDF、Word 格式"
            )
            
            if uploaded_files:
                st.success(f"已选择 {len(uploaded_files)} 个文件")
                
                # 显示文件列表
                with st.expander("📋 文件列表"):
                    for i, file in enumerate(uploaded_files, 1):
                        st.write(f"{i}. {file.name} ({file.size / 1024:.2f} KB)")
                
                # 构建知识库按钮
                col1, col2 = st.columns([1, 3])
                with col1:
                    build_button = st.button("🔨 构建知识库", type="primary", use_container_width=True)
                
                if build_button:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        if build_knowledge_base_from_files(uploaded_files, temp_dir):
                            st.rerun()
        
        else:  # 从文件系统选择
            st.markdown("**从文件系统选择文档：**")
            
            # 输入文档路径
            doc_path = st.text_input(
                "文档路径",
                value="sample_documents",
                help="输入文件路径或目录路径（相对于项目根目录）"
            )
            
            if doc_path:
                # 检查路径是否存在
                full_path = os.path.join(os.path.dirname(__file__), "..", doc_path)
                full_path = os.path.abspath(full_path)
                
                if os.path.exists(full_path):
                    if os.path.isdir(full_path):
                        # 列出目录中的文件
                        files = []
                        for ext in ['.txt', '.md', '.pdf', '.docx']:
                            files.extend(Path(full_path).glob(f"*{ext}"))
                            files.extend(Path(full_path).rglob(f"*{ext}"))
                        
                        if files:
                            st.success(f"找到 {len(files)} 个文档文件")
                            with st.expander("📋 文件列表"):
                                for i, file in enumerate(files[:20], 1):  # 最多显示20个
                                    st.write(f"{i}. {file.name}")
                            if len(files) > 20:
                                st.caption(f"... 还有 {len(files) - 20} 个文件")
                        else:
                            st.warning("目录中没有找到支持的文档文件")
                    else:
                        st.info(f"文件: {os.path.basename(full_path)}")
                    
                    # 构建按钮
                    if st.button("🔨 从路径构建知识库", type="primary"):
                        if build_knowledge_base_from_path(full_path):
                            st.rerun()
                else:
                    st.error(f"路径不存在: {full_path}")
                    st.caption("提示: 可以使用相对路径（如 'sample_documents'）或绝对路径")

# Tab 2: 问答
with tab2:
    st.header("💬 智能问答")
    
    if not st.session_state.kb_built:
        st.info("👆 请先在「文档管理」标签页上传文档并构建知识库")
    else:
        # 在问答页面显示阈值设置（更直观）
        col1, col2 = st.columns([2, 1])
        with col1:
            threshold = st.slider(
                "🎯 相关性阈值",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.similarity_threshold,
                step=0.05,
                help="过滤低相关性结果。值越高，只返回越相关的结果。\n"
                     "• 0.0: 不过滤（返回所有结果）\n"
                     "• 0.5: 中等严格（过滤相似度 < 50% 的结果）\n"
                     "• 0.7: 较严格（过滤相似度 < 70% 的结果）\n"
                     "• 0.8: 很严格（只返回相似度 >= 80% 的结果）"
            )
            if threshold != st.session_state.similarity_threshold:
                st.session_state.similarity_threshold = threshold
        
        with col2:
            if threshold == 0.0:
                st.metric("当前模式", "不过滤")
            elif threshold < 0.5:
                st.metric("当前模式", f"宽松 ({threshold:.0%})")
            elif threshold < 0.7:
                st.metric("当前模式", f"中等 ({threshold:.0%})")
            elif threshold < 0.8:
                st.metric("当前模式", f"严格 ({threshold:.0%})")
            else:
                st.metric("当前模式", f"很严格 ({threshold:.0%})")
        
        st.divider()
        
        # 显示对话历史
        if st.session_state.chat_history:
            st.subheader("💭 对话历史")
            for i, (role, content, sources) in enumerate(st.session_state.chat_history):
                if role == "user":
                    with st.chat_message("user"):
                        st.write(content)
                else:
                    with st.chat_message("assistant"):
                        st.write(content)
                        if sources:
                            with st.expander(f"📚 参考来源 ({len(sources)} 个)"):
                                for j, source in enumerate(sources, 1):
                                    st.markdown(f"**来源 {j}**: {source['metadata'].get('file_name', '未知')}")
                                    st.caption(f"相似度: {1/(1+source['distance']):.2%}")
                                    st.text(source['content'][:200] + "...")
            st.divider()
        
        # 问题输入
        question = st.chat_input("输入您的问题...")
        
        if question:
            # 添加用户问题到历史
            st.session_state.chat_history.append(("user", question, None))
            
            # 查询
            result = query_knowledge_base(question)
            
            if result:
                # 添加答案到历史
                st.session_state.chat_history.append(("assistant", result['answer'], result['sources']))
                st.rerun()

# Tab 3: 统计
with tab3:
    st.header("📊 系统统计")
    
    if st.session_state.qa_system and st.session_state.kb_built:
        stats = st.session_state.qa_system.get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("文档片段数", stats['total_documents'])
        with col2:
            st.metric("向量维度", stats['dimension'])
        with col3:
            st.metric("索引大小", stats['index_size'])
        
        st.divider()
        
        # 对话统计
        st.subheader("💬 对话统计")
        user_questions = sum(1 for role, _, _ in st.session_state.chat_history if role == "user")
        st.metric("已回答问题数", user_questions)
        
        if st.session_state.chat_history:
            st.subheader("📝 最近对话")
            for role, content, _ in st.session_state.chat_history[-5:]:
                if role == "user":
                    st.markdown(f"**用户**: {content}")
                else:
                    st.markdown(f"**助手**: {content[:100]}...")
    else:
        st.info("请先构建知识库以查看统计信息")

# 页脚
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>基于 LLaMA 的知识库问答系统 | 使用 Streamlit 构建</p>
</div>
""", unsafe_allow_html=True)

