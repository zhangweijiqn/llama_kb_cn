# 模型选择指南

## 默认模型

系统默认使用 **Qwen2-1.5B-Instruct** 模型：

- **大小**: 约 1.8GB
- **语言**: 中文优化
- **类型**: 指令跟随模型（Instruct）
- **特点**: 
  - ✅ 中文理解能力强
  - ✅ 资源占用低
  - ✅ 响应速度快
  - ✅ 无需特殊权限

## 模型下载

首次使用时，模型会自动从 HuggingFace 下载：

```bash
# 模型会下载到缓存目录
# Linux/Mac: ~/.cache/huggingface/
# Windows: C:\Users\<username>\.cache\huggingface\
```

如果下载失败，可以手动下载：

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载模型
huggingface-cli download Qwen/Qwen2-1.5B-Instruct
```

## 其他模型选项

### 1. Qwen 系列（推荐中文模型）

#### Qwen2-1.5B-Instruct（默认）
```python
qa_system = KnowledgeBaseQA(
    llama_model_name="Qwen/Qwen2-1.5B-Instruct"
)
```
- 大小: ~1.8GB
- 性能: ⭐⭐⭐⭐
- 中文: ✅ 优秀

#### Qwen-1.8B-Chat
```python
qa_system = KnowledgeBaseQA(
    llama_model_name="Qwen/Qwen-1.8B-Chat"
)
```
- 大小: ~1.8GB
- 性能: ⭐⭐⭐
- 中文: ✅ 良好

#### Qwen2-7B-Instruct（更强性能）
```python
qa_system = KnowledgeBaseQA(
    llama_model_name="Qwen/Qwen2-7B-Instruct",
    use_quantization=True  # 建议使用量化
)
```
- 大小: ~7GB（量化后约3.5GB）
- 性能: ⭐⭐⭐⭐⭐
- 中文: ✅ 优秀

### 2. ChatGLM 系列

#### ChatGLM3-6B（量化后）
```python
qa_system = KnowledgeBaseQA(
    llama_model_name="THUDM/chatglm3-6b",
    use_quantization=True  # 必须使用量化
)
```
- 大小: ~6GB（量化后约3GB）
- 性能: ⭐⭐⭐⭐
- 中文: ✅ 优秀

### 3. 使用量化版本

如果内存不足（< 8GB），建议使用量化：

```python
qa_system = KnowledgeBaseQA(
    llama_model_name="Qwen/Qwen2-1.5B-Instruct",
    use_quantization=True  # 8-bit 量化，节省约50%内存
)
```

**量化效果：**
- 内存占用减少约 50%
- 性能略有下降（通常 < 5%）
- 推理速度基本不变

## 模型对比

| 模型 | 大小 | 内存需求 | 中文支持 | 性能 | 推荐场景 |
|------|------|----------|----------|------|----------|
| Qwen2-1.5B-Instruct | 1.8GB | 4GB+ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 默认推荐 |
| Qwen-1.8B-Chat | 1.8GB | 4GB+ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 轻量级 |
| Qwen2-7B-Instruct | 7GB | 8GB+ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高性能需求 |
| ChatGLM3-6B | 6GB | 8GB+ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 专业场景 |

## 本地模型使用

### 下载模型到本地

```bash
# 使用 huggingface-cli
huggingface-cli download Qwen/Qwen2-1.5B-Instruct --local-dir ./models/qwen2-1.5b
```

### 使用本地模型

```python
qa_system = KnowledgeBaseQA(
    llama_model_name="./models/qwen2-1.5b"  # 本地路径
)
```

## 模型切换

### 在代码中切换

```python
# 使用不同的模型
qa_system = KnowledgeBaseQA(
    llama_model_name="Qwen/Qwen2-7B-Instruct",
    use_quantization=True
)
```

### 在 Web 界面中切换

修改 `web_app.py`：

```python
st.session_state.qa_system = KnowledgeBaseQA(
    llama_model_name="Qwen/Qwen2-7B-Instruct",  # 修改这里
    use_quantization=True
)
```

## 性能优化建议

### 1. 内存优化

- **4GB 内存**: 使用 Qwen2-1.5B-Instruct（默认）
- **8GB 内存**: 可以使用 Qwen2-7B-Instruct（量化）
- **16GB+ 内存**: 可以使用完整版大模型

### 2. 速度优化

- 使用 GPU 加速（自动检测）
- 减少 `max_length` 参数
- 使用量化版本

### 3. 质量优化

- 使用更大的模型（如 Qwen2-7B）
- 增加 `top_k` 检索数量
- 优化提示词

## 常见问题

### Q: 模型下载很慢？

A: 
1. 使用国内镜像（如果可用）
2. 手动下载到本地
3. 使用代理

### Q: 内存不足？

A:
1. 使用量化版本：`use_quantization=True`
2. 使用更小的模型
3. 关闭其他程序

### Q: 模型加载失败？

A:
1. 检查网络连接
2. 检查 HuggingFace 访问权限
3. 尝试手动下载模型
4. 检查磁盘空间

### Q: 如何更新模型？

A:
```bash
# 清除缓存
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct

# 重新运行，会自动下载最新版本
```

## 模型信息

### Qwen2-1.5B-Instruct

- **发布**: 2024年
- **参数量**: 1.5B
- **训练数据**: 多语言，中文优化
- **许可证**: Apache 2.0
- **HuggingFace**: https://huggingface.co/Qwen/Qwen2-1.5B-Instruct

### 技术细节

- **架构**: Transformer
- **上下文长度**: 32K tokens
- **支持格式**: ChatML, Instruct
- **量化支持**: 8-bit, 4-bit

