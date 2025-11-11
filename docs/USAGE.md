# 使用说明

## 快速开始

### 方式一：交互式界面（最简单）

```bash
python interactive_qa.py
```

然后按照提示操作：
```
> build sample_documents    # 构建知识库
> Python是什么？            # 直接提问
> stats                     # 查看统计
> quit                      # 退出
```

### 方式二：快速测试

```bash
python simple_test.py
```

这会自动创建示例文档并测试系统。

### 方式三：在代码中使用

```python
from kb_qa import KnowledgeBaseQA

# 初始化
qa = KnowledgeBaseQA(vector_store_path='my_kb')

# 构建知识库
qa.build_knowledge_base('./documents')

# 查询
result = qa.query("你的问题")
print(result['answer'])
```

## 详细功能说明

### 1. 文档加载

系统支持多种文档格式：
- **TXT**: 纯文本文件
- **Markdown**: Markdown 格式文档
- **PDF**: PDF 文档（需要 PyPDF2）
- **Word**: .docx 文件（需要 python-docx）

```python
from kb_qa import DocumentLoader

loader = DocumentLoader()

# 加载单个文件
doc = loader.load_document('document.txt')

# 加载整个目录
docs = loader.load_directory('./documents')

# 分割文本
chunks = loader.split_text(doc['content'], chunk_size=500, chunk_overlap=50)
```

### 2. 向量存储

向量存储使用 FAISS 进行高效检索：

```python
from kb_qa import VectorStore

# 初始化
store = VectorStore(embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2')

# 添加文档
store.add_documents(['文本1', '文本2'], [{'source': 'doc1'}, {'source': 'doc2'}])

# 搜索
results = store.search('查询文本', k=5)

# 保存
store.save()

# 加载
store.load()
```

### 3. LLaMA 模型

LLaMA 模型支持多种配置：

```python
from kb_qa import LLaMAModel

# 使用 HuggingFace 模型（需要访问权限）
llm = LLaMAModel(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    use_quantization=True  # 使用量化节省内存
)

# 生成文本
answer = llm.generate("你的问题", max_length=512, temperature=0.7)

# 对话模式
messages = [
    {"role": "user", "content": "你好"}
]
response = llm.chat(messages)
```

### 4. 完整问答系统

```python
from kb_qa import KnowledgeBaseQA

# 初始化
qa = KnowledgeBaseQA(
    embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
    llama_model_name="meta-llama/Llama-2-7b-chat-hf",
    use_quantization=True,
    vector_store_path='my_knowledge_base'
)

# 构建知识库
qa.build_knowledge_base(
    document_path='./documents',
    chunk_size=500,
    chunk_overlap=50
)

# 简单查询
result = qa.query("什么是机器学习？", top_k=3)

# 对话模式（支持上下文）
history = []
result1 = qa.chat("Python是什么？", conversation_history=history)
history.append({"role": "user", "content": "Python是什么？"})
history.append({"role": "assistant", "content": result1['answer']})
result2 = qa.chat("它有哪些特点？", conversation_history=history)
```

## 配置选项

### 嵌入模型选择

- `paraphrase-multilingual-MiniLM-L12-v2` (默认，支持中英文)
- `sentence-transformers/all-MiniLM-L6-v2` (英文，更快)
- `shibing624/text2vec-base-chinese` (中文优化)

### LLaMA 模型选择

**需要 HuggingFace 访问权限：**
- `meta-llama/Llama-2-7b-chat-hf` (7B，推荐)
- `meta-llama/Llama-2-13b-chat-hf` (13B，更强)

**无需权限的替代方案：**
- 使用本地模型路径
- 使用较小的开源模型（如 `gpt2`）进行测试

### 参数调优

- **chunk_size**: 文本块大小（默认 500）
  - 较小值：更精确的检索，但可能丢失上下文
  - 较大值：保留更多上下文，但检索可能不够精确

- **chunk_overlap**: 块重叠大小（默认 50）
  - 避免在块边界丢失信息

- **top_k**: 检索文档数量（默认 3-5）
  - 更多文档：更全面的答案，但可能包含不相关信息
  - 较少文档：更聚焦，但可能遗漏重要信息

- **temperature**: 生成温度（默认 0.7）
  - 较低值（0.3-0.5）：更确定、保守的答案
  - 较高值（0.8-1.0）：更创新、多样的答案

## 常见问题

### Q: 如何提高检索准确性？

A:
1. 优化文档分割参数（`chunk_size`, `chunk_overlap`）
2. 使用更合适的嵌入模型
3. 增加 `top_k` 值
4. 确保文档质量良好

### Q: 如何提高生成质量？

A:
1. 使用更大的 LLaMA 模型
2. 调整生成参数（`temperature`, `max_length`）
3. 优化提示词模板
4. 增加检索的文档数量

### Q: 内存不足怎么办？

A:
1. 启用量化：`use_quantization=True`
2. 使用 CPU 模式（自动检测）
3. 减少 `chunk_size` 和批次大小
4. 使用更小的模型

### Q: 如何添加新文档？

A:
```python
# 重新构建知识库（会合并新旧文档）
qa.build_knowledge_base('./documents')

# 或者手动添加
from kb_qa import DocumentLoader, VectorStore

loader = DocumentLoader()
doc = loader.load_document('new_doc.txt')
chunks = loader.split_text(doc['content'])
store.add_documents(chunks)
store.save()
```

## 性能优化建议

1. **批量处理**: 一次性加载多个文档，而不是逐个处理
2. **缓存向量**: 向量存储会自动保存，避免重复计算
3. **使用 GPU**: 如果有 GPU，系统会自动使用
4. **量化模型**: 使用 `use_quantization=True` 可以大幅减少内存使用

## 示例场景

### 场景1: 技术文档问答

```python
qa = KnowledgeBaseQA()
qa.build_knowledge_base('./tech_docs')
result = qa.query("如何配置数据库连接？")
```

### 场景2: 知识库助手

```python
qa = KnowledgeBaseQA()
qa.build_knowledge_base('./knowledge_base')

# 多轮对话
history = []
while True:
    question = input("问题: ")
    result = qa.chat(question, conversation_history=history)
    print(result['answer'])
    # 更新历史...
```

### 场景3: 文档检索系统

```python
# 仅使用检索功能，不使用 LLM
qa = KnowledgeBaseQA(llama_model_name='dummy')
qa.build_knowledge_base('./documents')

# 检索相关文档
result = qa.query("关键词", top_k=10)
for doc in result['sources']:
    print(doc['content'])
```

