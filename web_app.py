"""
基于 Streamlit 的知识库问答系统 Web 界面
"""
import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime
import hashlib
import re

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 页面配置（必须在第一个 Streamlit 命令中调用）
st.set_page_config(
    page_title="知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"  # 默认收起侧边栏
)

# 导入 KnowledgeBaseQA（必须在 st.set_page_config() 之后）
from kb_qa import KnowledgeBaseQA

# 使用 Streamlit 的 cache_resource 确保只初始化一次
@st.cache_resource
def get_global_qa_system():
    """获取全局的问答系统实例（使用 cache_resource 确保只初始化一次）"""
    print("=" * 60)
    print("正在初始化知识库问答系统（服务启动时加载一次）...")
    print("=" * 60)
    qa_system = KnowledgeBaseQA(
        vector_store_path='web_knowledge_base',
        llama_model_name="Qwen/Qwen2-0.5B-Instruct",  # 更小的中文模型，约500MB
        use_quantization=True,  # 使用量化以节省内存
        only_knowledge_base=True,  # 默认值，实际使用时从 session_state 获取
        similarity_threshold=0.0  # 初始化时不设置阈值，使用页面传入的值
    )
    print("=" * 60)
    print("知识库问答系统初始化完成")
    print("=" * 60)
    return qa_system

# 全局变量：知识库构建状态（所有会话共享）
_global_kb_built = False

# 服务启动时立即初始化系统（在 st.set_page_config() 之后调用）
# 使用 @st.cache_resource 确保只初始化一次，即使多次调用也不会重复初始化
_ = get_global_qa_system()  # 触发初始化，但不需要保存返回值

# 隐藏侧边栏
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="stSidebar"] + div {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

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
# qa_system 和 kb_built 现在使用全局变量，session_state 只用于存储用户设置
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'only_knowledge_base' not in st.session_state:
    st.session_state.only_knowledge_base = True  # 默认仅在知识库内回答
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.0  # 默认不过滤
if 'show_clear_history_confirm' not in st.session_state:
    st.session_state.show_clear_history_confirm = False
if 'show_clear_kb_confirm' not in st.session_state:
    st.session_state.show_clear_kb_confirm = False
if 'history_loaded' not in st.session_state:
    st.session_state.history_loaded = False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'show_change_password' not in st.session_state:
    st.session_state.show_change_password = False

# 文件路径
USERS_FILE = os.path.join('data', 'users.json')
CHAT_HISTORY_DIR = os.path.join('data', 'chat_histories')


def get_chat_history_file(username):
    """获取用户的对话历史文件路径"""
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    return os.path.join(CHAT_HISTORY_DIR, f'{username}_chat_history.json')


def hash_password(password):
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    """加载用户数据"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 创建默认管理员账号
            default_users = {
                'admin': {
                    'password': hash_password('admin'),
                    'password_plain': 'admin',  # 存储明文密码供管理员查看
                    'role': 'admin',
                    'status': 'approved',  # 已审批
                    'created_at': datetime.now().isoformat(),
                    'is_admin': True  # 标记为admin账号
                }
            }
            save_users(default_users)
            return default_users
    except Exception as e:
        print(f"加载用户数据失败: {e}")
        return {}


def save_users(users):
    """保存用户数据"""
    try:
        os.makedirs('data', exist_ok=True)
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存用户数据失败: {e}")
        return False


def authenticate(username, password):
    """验证用户登录"""
    users = load_users()
    if username in users:
        user = users[username]
        # 检查账号状态
        status = user.get('status', 'approved')  # 兼容旧数据，默认为已审批
        if status != 'approved':
            return False, None, status  # 返回状态信息
        if user['password'] == hash_password(password):
            return True, user['role'], status
    return False, None, None


def register_user(phone, password, role='user'):
    """注册新用户（使用手机号，需要管理员审批）"""
    users = load_users()
    
    # 验证手机号格式
    if not validate_phone(phone):
        return False, "手机号格式不正确（应为11位数字，以1开头）"
    
    if phone in users:
        return False, "该手机号已注册"
    
    users[phone] = {
        'password': hash_password(password),
        'password_plain': password,  # 存储明文密码供管理员查看
        'role': role,
        'status': 'pending',  # 待审批
        'created_at': datetime.now().isoformat(),
        'phone': phone  # 标记为手机号注册
    }
    
    if save_users(users):
        return True, "注册成功，等待管理员审批"
    else:
        return False, "注册失败"


def approve_user(username):
    """审批通过用户"""
    users = load_users()
    if username not in users:
        return False, "用户不存在"
    
    users[username]['status'] = 'approved'
    users[username]['approved_at'] = datetime.now().isoformat()
    users[username]['approved_by'] = st.session_state.current_user
    
    if save_users(users):
        return True, "账号已审批通过"
    else:
        return False, "审批失败"


def reject_user(username):
    """拒绝用户申请"""
    users = load_users()
    if username not in users:
        return False, "用户不存在"
    
    users[username]['status'] = 'rejected'
    users[username]['rejected_at'] = datetime.now().isoformat()
    users[username]['rejected_by'] = st.session_state.current_user
    
    if save_users(users):
        return True, "账号已拒绝"
    else:
        return False, "操作失败"


def delete_user(username):
    """删除用户"""
    users = load_users()
    if username not in users:
        return False, "用户不存在"
    
    # 不能删除自己
    if username == st.session_state.current_user:
        return False, "不能删除自己的账号"
    
    # 不能删除管理员账号
    if users[username].get('role') == 'admin':
        return False, "不能删除管理员账号"
    
    # 删除用户
    del users[username]
    
    # 删除用户的会话历史
    history_file = get_chat_history_file(username)
    if os.path.exists(history_file):
        try:
            os.remove(history_file)
        except:
            pass
    
    if save_users(users):
        return True, "账号已删除"
    else:
        return False, "删除失败"


def update_user_role(username, new_role):
    """修改用户角色"""
    users = load_users()
    if username not in users:
        return False, "用户不存在"
    
    # 不能修改自己的角色
    if username == st.session_state.current_user:
        return False, "不能修改自己的角色"
    
    # admin 账号是超级管理员，不能改为普通用户
    if username == 'admin' and new_role == 'user':
        return False, "admin 是超级管理员，不能设置为普通账户"
    
    # 不能将最后一个管理员改为普通用户
    if users[username].get('role') == 'admin' and new_role == 'user':
        # 计算除了当前要修改的账号之外，还有多少个已审批的管理员
        admin_count = sum(1 for u, info in users.items() 
                         if u != username  # 排除当前要修改的账号
                         and info.get('role') == 'admin' 
                         and info.get('status', 'approved') == 'approved')
        if admin_count < 1:  # 如果除了当前账号外没有其他管理员，就不允许
            return False, "不能将最后一个管理员改为普通用户"
    
    users[username]['role'] = new_role
    users[username]['role_updated_at'] = datetime.now().isoformat()
    users[username]['role_updated_by'] = st.session_state.current_user
    
    if save_users(users):
        return True, "角色已更新"
    else:
        return False, "更新失败"


def change_password(username, old_password, new_password):
    """修改密码（用户自己修改）"""
    users = load_users()
    if username not in users:
        return False, "用户不存在"
    
    # 验证旧密码
    if users[username]['password'] != hash_password(old_password):
        return False, "旧密码错误"
    
    # 更新密码
    users[username]['password'] = hash_password(new_password)
    users[username]['password_plain'] = new_password  # 存储明文密码供管理员查看
    users[username]['password_changed_at'] = datetime.now().isoformat()
    users[username]['password_changed_by'] = username
    
    if save_users(users):
        return True, "密码修改成功"
    else:
        return False, "密码修改失败"


def admin_reset_password(username, new_password):
    """管理员重置密码（不需要旧密码）"""
    users = load_users()
    if username not in users:
        return False, "用户不存在"
    
    # admin 超级管理员的密码只能由自己修改，其他管理员不能修改
    if username == 'admin' and st.session_state.current_user != 'admin':
        return False, "admin 超级管理员的密码只能由自己修改"
    
    # 更新密码
    users[username]['password'] = hash_password(new_password)
    users[username]['password_plain'] = new_password  # 存储明文密码供管理员查看
    users[username]['password_changed_at'] = datetime.now().isoformat()
    users[username]['password_changed_by'] = st.session_state.current_user
    
    if save_users(users):
        return True, "密码已重置"
    else:
        return False, "密码重置失败"


def validate_phone(phone):
    """验证手机号格式（11位数字）"""
    import re
    pattern = r'^1[3-9]\d{9}$'
    return bool(re.match(pattern, phone))


def super_admin_login(recovery_code):
    """超级管理员恢复登录"""
    # 超级管理员恢复码（可以配置）
    SUPER_ADMIN_CODE = "RECOVERY2024"  # 建议修改为更安全的代码
    
    if recovery_code == SUPER_ADMIN_CODE:
        users = load_users()
        if 'admin' in users:
            # 重置admin密码为默认密码
            users['admin']['password'] = hash_password('admin')
            users['admin']['password_plain'] = 'admin'
            users['admin']['recovery_used_at'] = datetime.now().isoformat()
            save_users(users)
            return True, "admin", "admin"
    return False, None, None


def save_chat_history():
    """保存对话历史到文件（按用户）"""
    if not st.session_state.current_user:
        return False
    
    try:
        # 确保目录存在
        os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
        
        # 保存对话历史
        history_file = get_chat_history_file(st.session_state.current_user)
        history_data = {
            'chat_history': st.session_state.chat_history,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"保存对话历史失败: {e}")
        return False


def load_chat_history():
    """从文件加载对话历史（按用户）"""
    if not st.session_state.current_user:
        return [], None
    
    try:
        history_file = get_chat_history_file(st.session_state.current_user)
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                return history_data.get('chat_history', []), history_data.get('last_updated', '')
        return [], None
    except Exception as e:
        print(f"加载对话历史失败: {e}")
        return [], None


def clear_chat_history_file():
    """清空对话历史文件（按用户）"""
    if not st.session_state.current_user:
        return False
    
    try:
        history_file = get_chat_history_file(st.session_state.current_user)
        if os.path.exists(history_file):
            os.remove(history_file)
        return True
    except Exception as e:
        print(f"清空对话历史文件失败: {e}")
        return False


# 初始化知识库状态（检查是否已有内容）
# 使用延迟初始化，在第一次调用时检查
def _init_kb_status():
    """初始化知识库状态（延迟执行，避免在模块级别执行）"""
    global _global_kb_built
    if not _global_kb_built:
        qa_system = get_global_qa_system()
        if qa_system and qa_system.has_content():
            _global_kb_built = True


def get_qa_system():
    """获取全局的问答系统实例（使用 cache_resource 确保只初始化一次）"""
    return get_global_qa_system()

def get_kb_built():
    """获取知识库构建状态"""
    global _global_kb_built
    return _global_kb_built

def set_kb_built(value):
    """设置知识库构建状态"""
    global _global_kb_built
    _global_kb_built = value

def check_knowledge_base_exists():
    """检查知识库是否存在"""
    qa_system = get_qa_system()
    if qa_system:
        return qa_system.has_content()
    return False


def clear_knowledge_base():
    """清空知识库"""
    global _global_kb_built
    qa_system = get_qa_system()
    if qa_system:
        try:
            qa_system.clear_knowledge_base()
            _global_kb_built = False
            # 清除缓存，强制重新初始化
            get_global_qa_system.clear()
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
    
    qa_system = get_qa_system()
    if qa_system is None:
        st.error("系统未初始化，请刷新页面重试")
        return False
    
    # 保存上传的文件到临时目录
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
    
    # 构建知识库
    qa_system = get_qa_system()
    if qa_system is None:
        st.error("系统未初始化，请刷新页面重试")
        return False
    
    with st.spinner("正在构建知识库，这可能需要一些时间..."):
        try:
            if len(saved_files) == 1:
                qa_system.build_knowledge_base(
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
                qa_system.build_knowledge_base(
                    multi_dir,
                    chunk_size=500,
                    chunk_overlap=50
                )
            
            set_kb_built(True)
            stats = qa_system.get_stats()
            # 显示构建完成的详细信息
            st.success("🎉 知识库构建完成！")
            st.balloons()  # 显示庆祝动画
            st.info(f"""
            **构建统计信息：**
            - 📄 文档片段数：{stats['total_documents']}
            - 📊 向量维度：{stats.get('dimension', 'N/A')}
            - 💾 索引大小：{stats.get('index_size', 'N/A')}
            
            ✅ 您现在可以开始使用知识库进行问答了！
            """)
            return True
        except Exception as e:
            st.error(f"构建失败: {e}")
            return False


def build_knowledge_base_from_path(document_path):
    """从文件路径构建知识库"""
    if not document_path or not os.path.exists(document_path):
        st.warning("请提供有效的文档路径！")
        return False
    
    qa_system = get_qa_system()
    if qa_system is None:
        st.error("系统未初始化，请刷新页面重试")
        return False
    
    # 构建知识库
    
    with st.spinner("正在构建知识库，这可能需要一些时间..."):
        try:
            qa_system.build_knowledge_base(
                document_path,
                chunk_size=500,
                chunk_overlap=50
            )
            
            set_kb_built(True)
            stats = qa_system.get_stats()
            # 显示构建完成的详细信息
            st.success("🎉 知识库构建完成！")
            st.balloons()  # 显示庆祝动画
            st.info(f"""
            **构建统计信息：**
            - 📄 文档片段数：{stats['total_documents']}
            - 📊 向量维度：{stats.get('dimension', 'N/A')}
            - 💾 索引大小：{stats.get('index_size', 'N/A')}
            
            ✅ 您现在可以开始使用知识库进行问答了！
            """)
            return True
        except Exception as e:
            st.error(f"构建失败: {e}")
            return False


def query_knowledge_base(question):
    """查询知识库"""
    if not get_kb_built():
        st.warning("请先构建知识库！")
        return None
    
    if not question.strip():
        return None
    
    qa_system = get_qa_system()
    if qa_system is None:
        st.warning("系统未初始化！")
        return None
    
    with st.spinner("正在检索和生成答案..."):
        try:
            result = qa_system.query(
                question,
                top_k=3,
                max_new_tokens=512,  # 增加生成长度，避免回答被截断
                temperature=0.7,
                similarity_threshold=st.session_state.similarity_threshold
            )
            return result
        except Exception as e:
            st.error(f"查询失败: {e}")
            return None


# 主界面
st.markdown('<div class="main-header">📚 知识库问答系统</div>', unsafe_allow_html=True)

# 登录/注册界面
if not st.session_state.logged_in:
    st.header("🔐 用户登录")
    
    login_tab, register_tab = st.tabs(["登录", "注册"])
    
    with login_tab:
        # 超级管理员恢复登录
        with st.expander("🔐 超级管理员恢复", expanded=False):
            st.info("如果忘记admin密码，可以使用恢复码重置")
            recovery_code = st.text_input("恢复码", type="password", key="recovery_code")
            if st.button("恢复登录", key="recovery_login"):
                if recovery_code:
                    success, username, password = super_admin_login(recovery_code)
                    if success:
                        st.success(f"恢复成功！admin密码已重置为: {password}")
                        st.info("请使用 admin / admin 登录")
                        st.rerun()
                    else:
                        st.error("恢复码错误")
                else:
                    st.warning("请输入恢复码")
        
        with st.form("login_form"):
            username = st.text_input("用户名/手机号", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            submit = st.form_submit_button("登录", use_container_width=True)
            
            if submit:
                if username and password:
                    success, role, status = authenticate(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.current_user = username
                        st.session_state.user_role = role
                        # 加载用户对话历史
                        saved_history, _ = load_chat_history()
                        st.session_state.chat_history = saved_history
                        st.session_state.history_loaded = True
                        st.success(f"登录成功！欢迎，{username} ({'管理员' if role == 'admin' else '普通用户'})")
                        st.rerun()
                    else:
                        if status == 'pending':
                            st.warning("账号待审批，请联系管理员审批后登录")
                        elif status == 'rejected':
                            st.error("账号已被拒绝，请联系管理员")
                        else:
                            st.error("用户名或密码错误")
                else:
                    st.warning("请输入用户名和密码")
    
    with register_tab:
        st.info("💡 请使用手机号注册（11位数字，以1开头）")
        with st.form("register_form"):
            phone = st.text_input("手机号", key="register_phone", help="请输入11位手机号")
            new_password = st.text_input("密码", type="password", key="register_password")
            confirm_password = st.text_input("确认密码", type="password", key="confirm_password")
            submit = st.form_submit_button("注册", use_container_width=True)
            
            if submit:
                if phone and new_password:
                    if new_password != confirm_password:
                        st.error("两次输入的密码不一致")
                    elif len(new_password) < 3:
                        st.warning("密码长度至少为3位")
                    else:
                        success, message = register_user(phone, new_password, role='user')
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    st.warning("请填写完整信息")
    
    st.info("💡 默认管理员账号：admin / admin")
    st.stop()

# 已登录用户界面
if st.session_state.logged_in:
    # 加载用户的历史对话（如果未加载）
    if not st.session_state.history_loaded:
        saved_history, last_updated = load_chat_history()
        if saved_history:
            st.session_state.chat_history = saved_history
        else:
            st.session_state.chat_history = []
        st.session_state.history_loaded = True
    
    # 显示用户信息和退出按钮
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        role_text = "管理员" if st.session_state.user_role == 'admin' else "普通用户"
        st.info(f"👤 当前用户: {st.session_state.current_user} ({role_text})")
    with col2:
        if st.button("🔑 修改密码", use_container_width=True):
            st.session_state.show_change_password = True
            st.rerun()
    with col3:
        if st.button("🔄 刷新会话", use_container_width=True):
            saved_history, _ = load_chat_history()
            st.session_state.chat_history = saved_history
            st.success("会话已刷新")
            st.rerun()
    with col4:
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.user_role = None
            st.session_state.chat_history = []
            st.session_state.history_loaded = False
            # 不清空 qa_system 和 kb_built，因为所有用户共享知识库
            st.rerun()
    
    # 修改密码对话框
    if st.session_state.show_change_password:
        with st.container():
            st.subheader("🔑 修改密码")
            with st.form("change_password_form"):
                old_password = st.text_input("旧密码", type="password", key="old_password")
                new_password = st.text_input("新密码", type="password", key="new_password")
                confirm_password = st.text_input("确认新密码", type="password", key="confirm_new_password")
                
                col1, col2 = st.columns(2)
                with col1:
                    submit = st.form_submit_button("✅ 确认修改", use_container_width=True, type="primary")
                with col2:
                    cancel = st.form_submit_button("❌ 取消", use_container_width=True)
                
                if cancel:
                    st.session_state.show_change_password = False
                    st.rerun()
                
                if submit:
                    if old_password and new_password and confirm_password:
                        if new_password != confirm_password:
                            st.error("两次输入的新密码不一致")
                        elif len(new_password) < 3:
                            st.warning("新密码长度至少为3位")
                        elif old_password == new_password:
                            st.warning("新密码不能与旧密码相同")
                        else:
                            success, message = change_password(
                                st.session_state.current_user,
                                old_password,
                                new_password
                            )
                            if success:
                                st.success(message)
                                st.session_state.show_change_password = False
                                st.rerun()
                            else:
                                st.error(message)
                    else:
                        st.warning("请填写完整信息")
    
    st.divider()

# 主内容区域
if st.session_state.logged_in:
    # 根据用户角色显示不同的标签页
    if st.session_state.user_role == 'admin':
        tab1, tab2 = st.tabs(["💬 问答", "⚙️ 设置与管理"])
    else:
        tab1, = st.tabs(["💬 问答"])
        tab2 = None

# Tab 1: 问答
if st.session_state.logged_in:
    with tab1:
        st.header("💬 智能问答")
        
        # 初始化知识库状态（延迟执行，只在第一次调用时检查）
        _init_kb_status()
        
        if not get_kb_built():
            if st.session_state.user_role == 'admin':
                st.info("👆 请先在「设置与管理」标签页上传文档并构建知识库")
            else:
                st.info("👆 请等待管理员构建知识库")
        else:
            # 对话历史滚动区域
            chat_container = st.container()
            with chat_container:
                if st.session_state.chat_history:
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
                else:
                    st.info("💡 开始对话吧！在下方输入您的问题。")
            
            # 回答设置（所有用户都可以设置）
            with st.expander("⚙️ 回答设置", expanded=False):
                st.subheader("⚙️ 回答设置")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 回答模式设置
                    st.markdown("**回答模式**")
                    only_kb = st.checkbox(
                        "仅在知识库内回答",
                        value=st.session_state.only_knowledge_base,
                        help="开启：仅基于知识库内容回答；关闭：可以回答知识库之外的问题"
                    )
                    
                    if only_kb != st.session_state.only_knowledge_base:
                        st.session_state.only_knowledge_base = only_kb
                        qa_system = get_qa_system()
                        if qa_system:
                            qa_system.only_knowledge_base = only_kb
                        st.info("回答模式已更新" if only_kb else "已允许回答知识库外的问题")
                
                with col2:
                    # 相关性阈值设置
                    st.markdown("**知识库相关性阈值**")
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
                
                # 显示当前设置状态
                col1, col2, col3 = st.columns(3)
                with col1:
                    mode_text = "仅在知识库内" if st.session_state.only_knowledge_base else "允许知识库外"
                    st.metric("回答模式", mode_text)
                with col2:
                    if threshold == 0.0:
                        threshold_text = "不过滤"
                    elif threshold < 0.5:
                        threshold_text = f"宽松 ({threshold:.0%})"
                    elif threshold < 0.7:
                        threshold_text = f"中等 ({threshold:.0%})"
                    elif threshold < 0.8:
                        threshold_text = f"严格 ({threshold:.0%})"
                    else:
                        threshold_text = f"很严格 ({threshold:.0%})"
                    st.metric("阈值模式", threshold_text)
                with col3:
                    st.metric("当前阈值", f"{threshold:.0%}")
                
                st.divider()
                
                # 清空对话历史
                st.markdown("**对话管理**")
                if st.button("🗑️ 清空对话历史", use_container_width=True, type="secondary"):
                    st.session_state.show_clear_history_confirm = True
                    st.rerun()
                
                # 清空对话历史确认对话框
                if st.session_state.show_clear_history_confirm:
                    with st.container():
                        st.warning("⚠️ 确认清空对话历史")
                        st.info("此操作将删除所有对话记录，且无法恢复。")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ 确认清空", type="primary", use_container_width=True, key="confirm_clear_history_qa"):
                                st.session_state.chat_history = []
                                clear_chat_history_file()  # 同时清空文件
                                st.session_state.show_clear_history_confirm = False
                                st.success("对话历史已清空")
                                st.rerun()
                        with col2:
                            if st.button("❌ 取消", use_container_width=True, key="cancel_clear_history_qa"):
                                st.session_state.show_clear_history_confirm = False
                                st.rerun()
            
            # 固定输入框
            question = st.chat_input("输入您的问题...")
            
            if question:
                # 添加用户问题到历史
                st.session_state.chat_history.append(("user", question, None))
                
                # 查询
                result = query_knowledge_base(question)
                
                if result:
                    # 添加答案到历史
                    st.session_state.chat_history.append(("assistant", result['answer'], result['sources']))
                    # 自动保存对话历史
                    save_chat_history()
                    st.rerun()

# Tab 2: 设置与管理（仅管理员可见）
if st.session_state.logged_in and tab2:
    with tab2:
        st.header("⚙️ 设置与管理")
        
        # 使用折叠面板组织内容
        with st.expander("📊 知识库管理", expanded=True):
            st.subheader("📊 知识库管理")
            
            # 清空知识库按钮
            qa_system = get_qa_system()
            if qa_system and qa_system.has_content():
                if st.button("🗑️ 清空知识库", type="secondary", use_container_width=True):
                    st.session_state.show_clear_kb_confirm = True
                    st.rerun()
            else:
                st.button("🗑️ 清空知识库", disabled=True, use_container_width=True)
            
            # 清空知识库确认对话框
            if st.session_state.show_clear_kb_confirm:
                with st.container():
                    st.error("⚠️ 确认清空知识库")
                    st.warning("此操作将删除所有知识库内容，包括向量索引和文档数据，且无法恢复。")
                    st.info("清空后需要重新上传文档并构建知识库才能继续使用。")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ 确认清空", type="primary", use_container_width=True, key="confirm_clear_kb"):
                            if clear_knowledge_base():
                                st.session_state.show_clear_kb_confirm = False
                                st.success("知识库已清空")
                                st.rerun()
                    with col2:
                        if st.button("❌ 取消", use_container_width=True, key="cancel_clear_kb"):
                            st.session_state.show_clear_kb_confirm = False
                            st.rerun()
            
            st.divider()
            
            # 知识库状态显示
            st.subheader("📊 知识库状态")
            qa_system = get_qa_system()
            if qa_system:
                if qa_system.has_content():
                    stats = qa_system.get_stats()
                    st.success("✓ 知识库已加载")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("文档片段数", stats['total_documents'])
                    with col2:
                        st.metric("向量维度", stats['dimension'])
                    with col3:
                        st.metric("索引大小", stats['index_size'])
                    
                    set_kb_built(True)
                else:
                    st.warning("知识库为空，请上传文档并构建知识库")
            else:
                st.info("系统已自动初始化")
        
        with st.expander("📤 文档上传", expanded=False):
            st.subheader("📤 文档上传")
            
            # 系统已自动初始化
            qa_system = get_qa_system()
            if qa_system:
                # 检查是否已有知识库
                if qa_system.has_content():
                    st.info("💡 检测到已存在的知识库，您可以选择使用现有知识库或重新构建")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ 使用现有知识库", type="primary", use_container_width=True, key="use_existing_kb"):
                            set_kb_built(True)
                            st.success("已加载现有知识库，可以直接开始问答！")
                            st.rerun()
                    with col2:
                        st.markdown("**或者添加/更新文档：**")
                    
                    st.divider()
                    
                    # 构建模式选择
                    build_mode = st.radio(
                        "构建模式",
                        ["补充知识库（追加新文档）", "重新构建（清空后重建）"],
                        help="补充模式：在现有知识库基础上添加新文档；重新构建：清空现有知识库后重新构建",
                        horizontal=True
                    )
                    is_replace_mode = build_mode == "重新构建（清空后重建）"
                    
                    if is_replace_mode:
                        st.warning("⚠️ 重新构建模式将清空现有知识库，所有已有文档将被删除！")
                    
                    st.divider()
                
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
                        
                        # 显示文件列表（不使用 expander，因为已经在 expander 内部）
                        st.markdown("**📋 文件列表：**")
                        for i, file in enumerate(uploaded_files, 1):
                            st.write(f"{i}. {file.name} ({file.size / 1024:.2f} KB)")
                        
                        # 构建知识库按钮
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            build_button = st.button("🔨 构建知识库", type="primary", use_container_width=True, key="build_kb_upload")
                        
                        if build_button:
                            # 如果是重新构建模式，先清空知识库
                            if is_replace_mode:
                                if not clear_knowledge_base():
                                    st.error("清空知识库失败，无法继续构建")
                                    st.stop()
                                st.info("已清空现有知识库，开始重新构建...")
                            
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
                                    # 显示文件列表（不使用 expander，因为已经在 expander 内部）
                                    st.markdown("**📋 文件列表：**")
                                    for i, file in enumerate(files[:20], 1):  # 最多显示20个
                                        st.write(f"{i}. {file.name}")
                                    if len(files) > 20:
                                        st.caption(f"... 还有 {len(files) - 20} 个文件")
                                else:
                                    st.warning("目录中没有找到支持的文档文件")
                            else:
                                st.info(f"文件: {os.path.basename(full_path)}")
                            
                            # 构建模式选择（文件系统模式）
                            build_mode_path = st.radio(
                                "构建模式",
                                ["补充知识库（追加新文档）", "重新构建（清空后重建）"],
                                help="补充模式：在现有知识库基础上添加新文档；重新构建：清空现有知识库后重新构建",
                                horizontal=True,
                                key="build_mode_path"
                            )
                            is_replace_mode_path = build_mode_path == "重新构建（清空后重建）"
                            
                            if is_replace_mode_path:
                                st.warning("⚠️ 重新构建模式将清空现有知识库，所有已有文档将被删除！")
                            
                            # 构建按钮
                            if st.button("🔨 从路径构建知识库", type="primary", key="build_kb_path"):
                                # 如果是重新构建模式，先清空知识库
                                if is_replace_mode_path:
                                    if not clear_knowledge_base():
                                        st.error("清空知识库失败，无法继续构建")
                                        st.stop()
                                    st.info("已清空现有知识库，开始重新构建...")
                                
                                if build_knowledge_base_from_path(full_path):
                                    st.rerun()
                        else:
                            st.error(f"路径不存在: {full_path}")
                            st.caption("提示: 可以使用相对路径（如 'sample_documents'）或绝对路径")
        
        with st.expander("👥 账号管理", expanded=False):
            st.subheader("👥 账号管理")
            
            # 获取所有用户
            users = load_users()
            
            # 待审批账号
            pending_users = {u: info for u, info in users.items() 
                           if info.get('status', 'approved') == 'pending'}
            
            if pending_users:
                st.warning(f"⚠️ 有 {len(pending_users)} 个账号待审批")
                st.markdown("### 📋 待审批账号")
                for username, user_info in pending_users.items():
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        with col1:
                            st.markdown(f"**{username}** ({user_info.get('role', 'user')})")
                            created_at = user_info.get('created_at', '')
                            if created_at:
                                try:
                                    dt = datetime.fromisoformat(created_at)
                                    st.caption(f"注册时间: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                                except:
                                    st.caption(f"注册时间: {created_at}")
                        with col2:
                            if st.button("✅ 通过", key=f"approve_{username}", use_container_width=True):
                                success, message = approve_user(username)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                        with col3:
                            if st.button("❌ 拒绝", key=f"reject_{username}", use_container_width=True):
                                success, message = reject_user(username)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                        with col4:
                            # 待审批账号中的管理员也不能删除
                            if user_info.get('role') == 'admin':
                                st.button("🗑️ 删除", key=f"delete_pending_{username}", use_container_width=True, 
                                        type="secondary", disabled=True, help="不能删除管理员账号")
                            else:
                                if st.button("🗑️ 删除", key=f"delete_pending_{username}", use_container_width=True, type="secondary"):
                                    success, message = delete_user(username)
                                    if success:
                                        st.success(message)
                                        st.rerun()
                                    else:
                                        st.error(message)
                        st.divider()
            else:
                st.success("✓ 没有待审批账号")
            
            st.divider()
            
            # 所有账号列表
            st.subheader("📋 所有账号")
            
            # 筛选选项
            filter_option = st.radio(
                "筛选",
                ["全部", "已审批", "待审批", "已拒绝"],
                horizontal=True
            )
            
            # 根据筛选条件显示账号
            filtered_users = {}
            if filter_option == "全部":
                filtered_users = users
            elif filter_option == "已审批":
                filtered_users = {u: info for u, info in users.items() 
                                if info.get('status', 'approved') == 'approved'}
            elif filter_option == "待审批":
                filtered_users = {u: info for u, info in users.items() 
                                if info.get('status', 'approved') == 'pending'}
            elif filter_option == "已拒绝":
                filtered_users = {u: info for u, info in users.items() 
                                if info.get('status', 'approved') == 'rejected'}
            
            if filtered_users:
                # 账号列表
                for username, user_info in filtered_users.items():
                    with st.container():
                        status = user_info.get('status', 'approved')
                        role = user_info.get('role', 'user')
                        
                        # 状态标签
                        if status == 'approved':
                            status_label = "✅ 已审批"
                        elif status == 'pending':
                            status_label = "⏳ 待审批"
                        else:
                            status_label = "❌ 已拒绝"
                        
                        col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
                        with col1:
                            st.markdown(f"**{username}**")
                            st.caption(f"角色: {role} | 状态: {status_label}")
                            created_at = user_info.get('created_at', '')
                            if created_at:
                                try:
                                    dt = datetime.fromisoformat(created_at)
                                    st.caption(f"注册: {dt.strftime('%Y-%m-%d %H:%M')}")
                                except:
                                    pass
                        
                        with col2:
                            if status == 'pending':
                                if st.button("✅ 通过", key=f"approve_list_{username}", use_container_width=True):
                                    success, message = approve_user(username)
                                    if success:
                                        st.success(message)
                                        st.rerun()
                                    else:
                                        st.error(message)
                        
                        with col3:
                            # 修改角色
                            if username != st.session_state.current_user:
                                current_role = user_info.get('role', 'user')
                                new_role = 'admin' if current_role == 'user' else 'user'
                                role_text = "设为管理员" if current_role == 'user' else "设为普通用户"
                                
                                # admin 账号是超级管理员，不能改为普通用户
                                if username == 'admin' and new_role == 'user':
                                    st.button(role_text, key=f"role_{username}", use_container_width=True, disabled=True, 
                                            help="admin 是超级管理员，不能设置为普通账户")
                                # 检查是否是将最后一个管理员改为普通用户
                                elif current_role == 'admin' and new_role == 'user':
                                    users = load_users()
                                    # 计算除了当前要修改的账号之外，还有多少个已审批的管理员
                                    admin_count = sum(1 for u, info in users.items() 
                                                     if u != username  # 排除当前要修改的账号
                                                     and info.get('role') == 'admin' 
                                                     and info.get('status', 'approved') == 'approved')
                                    if admin_count < 1:  # 如果除了当前账号外没有其他管理员，就不允许
                                        st.button(role_text, key=f"role_{username}", use_container_width=True, disabled=True, 
                                                help="不能将最后一个管理员改为普通用户")
                                    else:
                                        if st.button(role_text, key=f"role_{username}", use_container_width=True):
                                            success, message = update_user_role(username, new_role)
                                            if success:
                                                st.success(message)
                                                st.rerun()
                                            else:
                                                st.error(message)
                                else:
                                    if st.button(role_text, key=f"role_{username}", use_container_width=True):
                                        success, message = update_user_role(username, new_role)
                                        if success:
                                            st.success(message)
                                            st.rerun()
                                        else:
                                            st.error(message)
                        
                        with col4:
                            # 重置密码
                            # admin 超级管理员的密码只能由自己修改
                            if username == 'admin' and st.session_state.current_user != 'admin':
                                st.button("🔑 重置密码", key=f"reset_pwd_{username}", use_container_width=True, 
                                        disabled=True, help="admin 超级管理员的密码只能由自己修改")
                            else:
                                if st.button("🔑 重置密码", key=f"reset_pwd_{username}", use_container_width=True):
                                    # 使用会话状态存储要重置密码的用户名
                                    st.session_state[f'reset_pwd_user_{username}'] = True
                                    st.rerun()
                        
                        with col5:
                            # 删除账号
                            if username != st.session_state.current_user:
                                # 管理员账号不能删除
                                if user_info.get('role') == 'admin':
                                    st.button("🗑️ 删除", key=f"delete_{username}", use_container_width=True, 
                                            type="secondary", disabled=True, help="不能删除管理员账号")
                                else:
                                    if st.button("🗑️ 删除", key=f"delete_{username}", use_container_width=True, type="secondary"):
                                        success, message = delete_user(username)
                                        if success:
                                            st.success(message)
                                            st.rerun()
                                        else:
                                            st.error(message)
                            else:
                                st.caption("当前账号")
                        
                        with col6:
                            # 查看详情
                            # admin 超级管理员的详情只能由自己查看
                            if username == 'admin' and st.session_state.current_user != 'admin':
                                st.button("📋 详情", key=f"detail_btn_{username}", use_container_width=True, 
                                        disabled=True, help="admin 超级管理员的详情只能由自己查看")
                            else:
                                if st.button("📋 详情", key=f"detail_btn_{username}", use_container_width=True):
                                    # admin 超级管理员的详情只能由自己查看
                                    if username == 'admin' and st.session_state.current_user != 'admin':
                                        st.error("admin 超级管理员的详情只能由自己查看")
                                    else:
                                        # 创建详情显示，将password字段替换为password_plain
                                        detail_info = user_info.copy()
                                        if 'password_plain' in detail_info:
                                            # 先保存原始的password（哈希值）
                                            detail_info['password_hash'] = detail_info.get('password', 'N/A')
                                            # 然后将password替换为明文密码
                                            detail_info['password'] = detail_info['password_plain']
                                            # 删除password_plain字段避免重复
                                            del detail_info['password_plain']
                                        else:
                                            # 如果没有password_plain，说明是旧数据，password就是哈希值
                                            detail_info['password_hash'] = detail_info.get('password', 'N/A')
                                            detail_info['password'] = '（哈希值，无明文）'
                                        st.json(detail_info)
                        
                        # 重置密码对话框
                        if st.session_state.get(f'reset_pwd_user_{username}', False):
                            with st.container():
                                st.markdown("---")
                                st.markdown(f"### 🔑 重置密码: {username}")
                                with st.form(f"reset_pwd_form_{username}"):
                                    new_pwd = st.text_input("新密码", type="password", key=f"new_pwd_{username}")
                                    confirm_pwd = st.text_input("确认新密码", type="password", key=f"confirm_pwd_{username}")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        submit_reset = st.form_submit_button("✅ 确认重置", use_container_width=True, type="primary")
                                    with col2:
                                        cancel_reset = st.form_submit_button("❌ 取消", use_container_width=True)
                                    
                                    if cancel_reset:
                                        st.session_state[f'reset_pwd_user_{username}'] = False
                                        st.rerun()
                                    
                                    if submit_reset:
                                        if new_pwd and confirm_pwd:
                                            if new_pwd != confirm_pwd:
                                                st.error("两次输入的密码不一致")
                                            elif len(new_pwd) < 3:
                                                st.warning("密码长度至少为3位")
                                            else:
                                                success, message = admin_reset_password(username, new_pwd)
                                                if success:
                                                    st.success(message)
                                                    st.session_state[f'reset_pwd_user_{username}'] = False
                                                    st.rerun()
                                                else:
                                                    st.error(message)
                                        else:
                                            st.warning("请填写完整信息")
                        
                        st.divider()
            else:
                st.info("没有符合条件的账号")
        
        with st.expander("📊 系统统计", expanded=False):
            st.subheader("📊 系统统计")
            
            qa_system = get_qa_system()
            if qa_system and get_kb_built():
                stats = qa_system.get_stats()
                
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

