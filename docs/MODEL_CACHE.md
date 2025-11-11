# 模型缓存目录说明

## 默认缓存位置

HuggingFace 模型默认下载到以下目录：

### macOS / Linux
```
~/.cache/huggingface/hub/
```

完整路径示例：
```
/Users/zhihu/.cache/huggingface/hub/
```

### Windows
```
C:\Users\<用户名>\.cache\huggingface\hub\
```

## 查看已下载的模型

### 方法1：使用检查脚本（推荐）

```bash
python scripts/check_models.py
```

这个脚本会显示：
- 缓存目录位置
- 已下载的模型列表
- 每个模型的大小和路径

### 方法2：手动查看

```bash
# macOS/Linux
ls -lh ~/.cache/huggingface/hub/

# 或直接打开目录
open ~/.cache/huggingface/hub/  # macOS
nautilus ~/.cache/huggingface/hub/  # Linux (GNOME)
```

### 方法3：使用 Python 代码

```python
from transformers import file_utils
import os

# 获取缓存目录
cache_dir = file_utils.default_cache_path
print(f"缓存目录: {cache_dir}")

# 查看模型
hub_dir = os.path.join(cache_dir, "hub")
if os.path.exists(hub_dir):
    models = [d for d in os.listdir(hub_dir) if d.startswith("models--")]
    for model in models:
        model_name = model.replace("models--", "").replace("--", "/")
        print(f"模型: {model_name}")
```

## 模型目录结构

下载的模型会按照以下结构存储：

```
~/.cache/huggingface/hub/
└── models--Qwen--Qwen2-0.5B-Instruct/
    ├── snapshots/
    │   └── <commit-hash>/
    │       ├── config.json
    │       ├── tokenizer.json
    │       ├── model.safetensors
    │       └── ...
    └── refs/
        └── main
```

## 自定义缓存目录

### 方法1：设置环境变量

```bash
# 设置 HuggingFace 缓存目录
export HF_HOME=/path/to/your/cache

# 或设置 transformers 缓存
export TRANSFORMERS_CACHE=/path/to/your/cache
```

### 方法2：在代码中指定

```python
from kb_qa import KnowledgeBaseQA

# 方法1: 设置环境变量（在代码中）
import os
os.environ['HF_HOME'] = '/path/to/your/cache'

qa_system = KnowledgeBaseQA()

# 方法2: 直接修改模型加载代码
# 需要修改 kb_qa/llama_model.py 中的 from_pretrained 调用
```

### 方法3：使用本地模型

```python
# 先下载到本地目录
# huggingface-cli download Qwen/Qwen2-0.5B-Instruct --local-dir ./models/qwen2-0.5b

# 然后使用本地路径
qa_system = KnowledgeBaseQA(
    llama_model_name="./models/qwen2-0.5b"
)
```

## 清理缓存

### 删除特定模型

```bash
# 删除 Qwen2-0.5B-Instruct 模型
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct
```

### 删除所有模型

```bash
# 清空整个缓存目录
rm -rf ~/.cache/huggingface/hub/*
```

### 查看缓存大小

```bash
# macOS/Linux
du -sh ~/.cache/huggingface/hub/

# 查看各个模型大小
du -sh ~/.cache/huggingface/hub/models--*
```

## 常见问题

### Q: 如何知道模型是否已下载？

A: 运行检查脚本：
```bash
python scripts/check_models.py
```

### Q: 模型下载到了哪里？

A: 默认在 `~/.cache/huggingface/hub/`，可以通过环境变量 `HF_HOME` 修改。

### Q: 如何移动模型到其他位置？

A:
```bash
# 1. 设置新的缓存目录
export HF_HOME=/new/path/to/cache

# 2. 或复制现有模型
cp -r ~/.cache/huggingface/hub/models--* /new/path/to/cache/hub/
```

### Q: 如何节省磁盘空间？

A:
1. 删除不需要的模型
2. 使用更小的模型（如 0.5B）
3. 使用量化版本
4. 定期清理缓存

### Q: 模型可以共享吗？

A: 可以，但需要注意：
- 确保 Python 版本兼容
- 确保 transformers 版本兼容
- 确保模型文件完整

## 快速命令

```bash
# 查看缓存目录
echo $HF_HOME
# 或
python -c "from transformers import file_utils; print(file_utils.default_cache_path)"

# 查看缓存大小
du -sh ~/.cache/huggingface/

# 列出所有模型
ls ~/.cache/huggingface/hub/models--*

# 删除特定模型
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct

# 清空所有缓存
rm -rf ~/.cache/huggingface/hub/*
```


