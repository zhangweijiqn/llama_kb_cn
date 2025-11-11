# 基于 LLaMA 的个人知识库问答系统

这是一个完整的基于开源 LLaMA 模型的个人知识库问答系统，支持文档加载、向量检索和 RAG（检索增强生成）问答。

## 功能特性

- 📚 **多格式文档支持**: 支持 PDF、Word、TXT、Markdown 等格式
- 🔍 **智能检索**: 基于向量相似度的语义检索
- 🤖 **LLaMA 集成**: 支持使用 LLaMA 模型进行问答生成
- 💾 **向量存储**: 使用 FAISS 进行高效的向量存储和检索
- 🔄 **RAG 架构**: 检索增强生成，结合知识库和 LLM
- ⚡ **量化支持**: 支持 8-bit 量化以节省内存

## 项目结构

```
.
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖包列表
├── .gitignore               # Git 忽略文件
├── kb_qa/                   # 核心模块包
│   ├── __init__.py          # 包初始化文件
│   ├── document_loader.py   # 文档加载和预处理模块
│   ├── vector_store.py      # 向量数据库模块
│   ├── llama_model.py       # LLaMA 模型加载和推理模块
│   └── knowledge_base_qa.py # 知识库问答系统主模块
├── tests/                   # 测试脚本
│   ├── simple_test.py       # 快速测试脚本
│   └── test_qa_system.py     # 完整测试脚本
├── examples/                # 示例脚本
│   └── example_usage.py      # 使用示例
├── scripts/                 # 工具脚本
│   ├── interactive_qa.py    # 交互式命令行界面
│   ├── quick_start.sh       # 快速启动脚本
│   ├── install.sh          # 依赖安装脚本
│   └── run_web.sh          # Web 应用启动脚本
├── web_app.py              # Streamlit Web 应用
├── docs/                    # 文档
│   └── USAGE.md             # 详细使用说明
└── data/                    # 数据存储目录（自动创建）
    ├── *.index              # FAISS 向量索引文件
    └── *.pkl                # 文本内容和元数据文件
```

**说明：**
- `data/` 目录用于存储向量索引和数据文件，由系统自动创建
- `.index` 文件：FAISS 向量索引，存储文档的向量索引用于快速检索
- `.pkl` 文件：Python pickle 格式，存储文本内容和元数据
- 这些文件用于持久化向量存储，避免每次重新计算，**非常重要**

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（可选，用于 GPU 加速）

## 安装步骤

### 1. 克隆或下载项目

```bash
cd /Users/zhihu/Projects/cursor
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 LLaMA 模型（可选）

#### 方式一：使用默认中文模型（推荐）

系统默认使用 **Qwen2-0.5B-Instruct** 模型：
- ✅ 中文优化，适合中文问答
- ✅ 模型大小约 500MB，资源占用极低
- ✅ 无需特殊权限，可直接下载使用
- ✅ 支持对话格式
- ✅ 适合内存受限环境

首次运行时会自动从 HuggingFace 下载模型（需要网络连接）。

#### 方式二：使用其他模型

可以在代码中指定其他模型：

```python
# 使用其他 Qwen 模型
qa_system = KnowledgeBaseQA(
    llama_model_name="Qwen/Qwen2-1.5B-Instruct"  # 更大的模型（约1.8GB）
)

# 使用量化版本（进一步节省内存）
qa_system = KnowledgeBaseQA(
    llama_model_name="Qwen/Qwen2-0.5B-Instruct",
    use_quantization=True  # 8-bit 量化（默认已启用）
)
```

#### 方式三：使用本地模型

将下载的模型放在本地目录，然后在代码中指定路径：

```python
qa_system = KnowledgeBaseQA(
    llama_model_name="./models/qwen2-1.5b"  # 本地路径
)
```

## 使用方法

### 快速测试（不使用 LLaMA，仅测试检索功能）

```bash
cd tests
python simple_test.py
```

或者从项目根目录：

```bash
python tests/simple_test.py
```

这个脚本会：
1. 创建示例文档
2. 构建知识库
3. 测试检索功能

### 完整测试

```bash
cd tests
python test_qa_system.py
```

### 模型加载测试（独立测试）

单独测试模型是否能正常加载：

```bash
# 基本测试（默认模型）
python scripts/test_model_load.py

# 使用量化模式
python scripts/test_model_load.py --quantization

# 测试其他模型
python scripts/test_model_load.py --model Qwen/Qwen2-1.5B-Instruct

# 跳过推理测试（仅测试加载）
python scripts/test_model_load.py --skip-inference
```

这个脚本会逐步测试：
1. 依赖导入
2. 模型配置加载
3. Tokenizer 加载
4. 模型加载
5. 模型推理（可选）

### Web 界面使用（推荐）

启动交互式 Web 界面：

```bash
# 方式1: 使用启动脚本
./scripts/run_web.sh

# 方式2: 直接运行
streamlit run web_app.py
```

浏览器会自动打开，如果没有，请访问: http://localhost:8501

**Web 界面功能：**
- 📄 文档上传和管理（支持多文件）
- 🔨 一键构建知识库
- 💬 智能问答对话
- 📚 答案来源追溯
- 📊 系统统计信息

### 命令行交互式使用

```bash
cd scripts
python interactive_qa.py
```

或者从项目根目录：

```bash
python scripts/interactive_qa.py
```

启动后可以：
- 输入 `build <文档路径>` 构建知识库
- 直接输入问题开始问答
- 输入 `stats` 查看统计信息
- 输入 `help` 查看帮助
- 输入 `quit` 退出

### 在代码中使用

```python
from kb_qa import KnowledgeBaseQA

# 初始化系统
qa_system = KnowledgeBaseQA(
    embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
    llama_model_name="meta-llama/Llama-2-7b-chat-hf",  # 或使用本地模型路径
    use_quantization=True,  # 使用量化以节省内存
    vector_store_path='my_knowledge_base'
)

# 构建知识库
qa_system.build_knowledge_base(
    document_path='./documents',  # 文档目录或文件路径
    chunk_size=500,              # 文本块大小
    chunk_overlap=50             # 块重叠大小
)

# 查询
result = qa_system.query(
    question="什么是人工智能？",
    top_k=3,                     # 检索的文档数量
    max_length=512,              # 生成的最大长度
    temperature=0.7              # 生成温度
)

print(result['answer'])
print(f"来源: {result['sources']}")
```

### 对话模式

```python
# 支持上下文对话
conversation_history = [
    {"role": "user", "content": "Python是什么？"},
    {"role": "assistant", "content": "Python是一种编程语言..."}
]

result = qa_system.chat(
    question="它有哪些特点？",
    conversation_history=conversation_history
)
```

## 模块说明

### 1. DocumentLoader (`document_loader.py`)

负责加载和预处理文档：
- 支持多种格式（PDF、Word、TXT、Markdown）
- 智能文本分割
- 保持文档元数据

### 2. VectorStore (`vector_store.py`)

向量数据库管理：
- 使用 FAISS 进行高效检索
- 支持持久化存储
- 基于语义相似度的检索

### 3. LLaMAModel (`llama_model.py`)

LLaMA 模型封装：
- 自动模型加载
- 支持量化（8-bit）
- 文本生成和对话接口

### 4. KnowledgeBaseQA (`knowledge_base_qa.py`)

主系统模块：
- 整合所有组件
- 实现 RAG 流程
- 提供简洁的 API

## 配置说明

### 嵌入模型选择

系统默认使用 `paraphrase-multilingual-MiniLM-L12-v2`，支持中英文。其他可选模型：
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（多语言）
- `sentence-transformers/all-MiniLM-L6-v2`（英文，更快）
- `shibing624/text2vec-base-chinese`（中文优化）

### 模型选择

**默认模型（推荐）：**
- `Qwen/Qwen2-0.5B-Instruct`（约500MB，中文优化，默认使用，最小）

**其他中文模型选项：**
- `Qwen/Qwen2-1.5B-Instruct`（约1.8GB，更强性能）
- `Qwen/Qwen-1.8B-Chat`（约1.8GB，中文对话）
- `Qwen/Qwen2-7B-Instruct`（约7GB，最强性能，需要量化）

**使用量化版本：**
如果内存不足，可以启用 8-bit 量化：
```python
qa_system = KnowledgeBaseQA(use_quantization=True)
```

### 内存优化

如果内存有限，可以：
1. 使用量化：`use_quantization=True`
2. 使用较小的模型
3. 减少 `chunk_size` 和 `top_k` 参数

## 常见问题

### Q: 模型下载失败怎么办？

A: 
1. 检查网络连接
2. 如果使用 HuggingFace，确保已登录：`huggingface-cli login`
3. 尝试使用本地模型路径
4. 使用较小的替代模型进行测试

### Q: 内存不足怎么办？

A:
1. 启用量化：`use_quantization=True`
2. 使用 CPU 模式（自动检测）
3. 减少 `chunk_size` 和批次大小
4. 使用更小的模型

### Q: 检索结果不准确？

A:
1. 调整 `top_k` 参数
2. 优化文档分割参数（`chunk_size`, `chunk_overlap`）
3. 尝试不同的嵌入模型
4. 确保文档质量良好

### Q: 生成答案质量不高？

A:
1. 增加检索的文档数量（`top_k`）
2. 调整生成参数（`temperature`, `max_length`）
3. 使用更大的 LLaMA 模型
4. 优化提示词模板

## 测试结果

运行 `simple_test.py` 应该看到类似输出：

```
============================================================
知识库问答系统 - 快速测试
============================================================

1. 创建示例文档...
✓ 示例文档已创建: sample_documents/测试文档.txt

2. 初始化系统（检索模式）...
✓ 系统初始化完成

3. 构建知识库...
✓ 知识库构建完成

知识库统计: {'total_documents': 5, 'index_size': 5, 'dimension': 384}

4. 测试查询功能...
============================================================

问题 1: Python是什么？
------------------------------------------------------------
答案:
根据知识库检索，最相关的信息如下：

Python是一种高级编程语言，由Guido van Rossum在1991年首次发布。

检索到 2 个相关文档片段
最相关片段: Python是一种高级编程语言，由Guido van Rossum在1991年首次发布。

Python的特点：
- 语法简洁清晰
- 易于学习和使用...
```

## 许可证

本项目基于开源 LLaMA 模型，请遵守相应的使用条款。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更多文档

- [详细使用说明](docs/USAGE.md) - 完整的功能说明和使用示例
- [Web 界面使用指南](docs/WEB_APP.md) - Web 应用的详细使用说明
- [模型选择指南](docs/MODEL_GUIDE.md) - 模型选择、配置和优化指南
- [模型缓存说明](docs/MODEL_CACHE.md) - 模型下载位置和缓存管理
- [故障排除指南](docs/TROUBLESHOOTING.md) - 常见问题解决方案（包括 segmentation fault）

## 更新日志

- 2024: 初始版本发布
  - ✅ 支持多格式文档加载（PDF、Word、TXT、Markdown）
  - ✅ 集成 FAISS 向量检索
  - ✅ 支持 LLaMA 模型问答
  - ✅ 实现 RAG 架构
  - ✅ 交互式命令行界面
  - ✅ 完整的测试和示例代码
