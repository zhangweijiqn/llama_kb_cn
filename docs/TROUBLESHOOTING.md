# 故障排除指南

## Segmentation Fault 错误

### 问题描述

在初始化系统时遇到 `segmentation fault` 错误，程序异常退出。

### 常见原因

1. **内存不足**：模型加载时内存耗尽
2. **依赖版本不兼容**：PyTorch/transformers 版本问题
3. **系统资源限制**：macOS 内存限制
4. **模型文件损坏**：下载的模型文件不完整

### 解决方案

#### 方案1：使用量化版本（推荐）

```python
from kb_qa import KnowledgeBaseQA

# 使用量化版本，减少内存占用
qa_system = KnowledgeBaseQA(
    use_quantization=True  # 启用 8-bit 量化
)
```

#### 方案2：跳过模型加载（仅使用检索）

```bash
# 设置环境变量
export SKIP_LLM_MODEL=true

# 然后运行
python your_script.py
```

或在代码中：

```python
qa_system = KnowledgeBaseQA(
    llama_model_name="dummy"  # 跳过模型加载
)
```

#### 方案3：使用更小的模型

```python
qa_system = KnowledgeBaseQA(
    llama_model_name="Qwen/Qwen-1.8B-Chat"  # 更小的模型
)
```

#### 方案4：检查并更新依赖

```bash
# 更新 PyTorch 和 transformers
pip install --upgrade torch transformers

# 检查版本兼容性
python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}')"
```

#### 方案5：增加系统资源

**macOS:**
- 关闭其他应用程序
- 增加虚拟内存
- 使用 Activity Monitor 检查内存使用

**Linux:**
```bash
# 检查内存
free -h

# 增加 swap（如果内存不足）
sudo swapon --show
```

#### 方案6：使用 GPU（如果有）

```python
# 系统会自动检测 GPU
# 确保安装了 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 调试方法

#### 1. 检查内存使用

```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

#### 2. 逐步加载

```python
# 先只加载 tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
print("Tokenizer 加载成功")

# 再加载模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
print("模型加载成功")
```

#### 3. 使用 gdb 调试（Linux）

```bash
# 安装 gdb
sudo apt-get install gdb

# 使用 gdb 运行
gdb python
(gdb) run your_script.py
# 当 segfault 发生时，查看堆栈
(gdb) bt
```

#### 4. 检查系统日志

**macOS:**
```bash
# 查看崩溃报告
ls ~/Library/Logs/DiagnosticReports/
```

**Linux:**
```bash
# 查看系统日志
dmesg | tail -20
journalctl -xe
```

### 预防措施

1. **使用虚拟环境**：避免依赖冲突
2. **定期更新依赖**：保持最新稳定版本
3. **监控资源使用**：在加载前检查可用内存
4. **使用量化**：默认启用量化以节省内存

### 快速修复脚本

创建 `fix_segfault.py`:

```python
#!/usr/bin/env python
"""快速修复 segmentation fault 问题"""

import os
import sys

def main():
    print("=" * 60)
    print("Segmentation Fault 修复工具")
    print("=" * 60)
    
    print("\n1. 检查环境...")
    try:
        import torch
        import transformers
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ 依赖缺失: {e}")
        print("运行: pip install torch transformers")
        return
    
    print("\n2. 测试模型加载（使用量化）...")
    try:
        from kb_qa import KnowledgeBaseQA
        qa = KnowledgeBaseQA(
            llama_model_name="Qwen/Qwen2-1.5B-Instruct",
            use_quantization=True  # 使用量化
        )
        print("✓ 模型加载成功（量化模式）")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        print("\n建议:")
        print("  1. 设置环境变量: export SKIP_LLM_MODEL=true")
        print("  2. 使用更小的模型")
        print("  3. 检查系统内存")
        return
    
    print("\n3. 测试检索功能...")
    try:
        # 测试检索（不需要模型）
        qa = KnowledgeBaseQA(llama_model_name="dummy")
        print("✓ 检索功能正常")
    except Exception as e:
        print(f"✗ 检索测试失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✓ 系统检查完成")
    print("=" * 60)
    print("\n如果仍有问题，请:")
    print("  1. 查看完整错误信息")
    print("  2. 检查系统资源")
    print("  3. 尝试使用 SKIP_LLM_MODEL=true")

if __name__ == "__main__":
    main()
```

运行：
```bash
python fix_segfault.py
```

## 其他常见问题

### 问题：模型下载很慢

**解决方案：**
```bash
# 使用镜像（如果可用）
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载
huggingface-cli download Qwen/Qwen2-1.5B-Instruct --local-dir ./models/qwen2-1.5b
```

### 问题：CUDA out of memory

**解决方案：**
```python
# 使用量化
qa_system = KnowledgeBaseQA(use_quantization=True)

# 或使用 CPU
qa_system = KnowledgeBaseQA()
# 确保 torch.cuda.is_available() 返回 False
```

### 问题：依赖版本冲突

**解决方案：**
```bash
# 创建新的虚拟环境
python -m venv venv_fresh
source venv_fresh/bin/activate  # Linux/Mac
# venv_fresh\Scripts\activate  # Windows

# 重新安装
pip install -r requirements.txt
```

## 获取帮助

如果问题仍未解决：

1. 查看完整错误堆栈
2. 检查系统资源（内存、磁盘）
3. 查看项目 Issues
4. 提供以下信息：
   - Python 版本
   - PyTorch 版本
   - Transformers 版本
   - 系统信息（OS、内存）
   - 完整错误信息

