# 依赖安装指南

## 快速安装

### 方式一：直接安装（推荐）

```bash
# 在项目根目录执行
pip install -r requirements.txt
```

### 方式二：使用虚拟环境（推荐用于生产环境）

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 方式三：使用 conda（如果使用 conda）

```bash
# 创建 conda 环境
conda create -n kb_qa python=3.8
conda activate kb_qa

# 安装依赖
pip install -r requirements.txt
```

## 分步安装（如果遇到问题）

如果一次性安装失败，可以分步安装：

```bash
# 1. 安装基础依赖
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install sentence-transformers>=2.2.0

# 2. 安装向量数据库
pip install faiss-cpu>=1.7.4

# 3. 安装其他依赖
pip install numpy>=1.24.0 pandas>=2.0.0
pip install PyPDF2>=3.0.0 python-docx>=1.0.0
pip install python-dotenv>=1.0.0
```

## 最小化安装（仅核心功能）

如果只需要基本功能，可以只安装核心依赖：

```bash
pip install torch transformers sentence-transformers faiss-cpu numpy
```

## 常见问题解决

### 1. 网络问题（使用国内镜像）

```bash
# 使用清华镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里云镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 2. 权限问题

```bash
# 使用 --user 参数
pip install -r requirements.txt --user
```

### 3. 升级 pip

```bash
python -m pip install --upgrade pip
```

### 4. 特定包安装失败

如果某个包安装失败，可以：

```bash
# 跳过失败的包，先安装其他的
pip install -r requirements.txt --ignore-installed <包名>

# 或单独安装特定版本
pip install <包名>==<版本号>
```

### 5. PyTorch 安装（根据系统选择）

**CPU 版本（默认）：**
```bash
pip install torch torchvision torchaudio
```

**GPU 版本（CUDA）：**
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 6. bitsandbytes 安装问题

`bitsandbytes` 主要用于模型量化，如果不需要可以跳过：

```bash
# 安装时排除 bitsandbytes
pip install -r requirements.txt --ignore-installed bitsandbytes
```

或者编辑 `requirements.txt`，注释掉这一行。

## 验证安装

安装完成后，验证关键包是否安装成功：

```bash
python -c "
import torch
import transformers
import sentence_transformers
import faiss
print('✓ 所有核心依赖安装成功！')
print(f'PyTorch 版本: {torch.__version__}')
print(f'Transformers 版本: {transformers.__version__}')
"
```

## 依赖说明

### 核心依赖（必需）

- **torch**: PyTorch 深度学习框架
- **transformers**: HuggingFace 模型库
- **sentence-transformers**: 文本嵌入模型
- **faiss-cpu**: 向量检索库（CPU 版本）

### 可选依赖

- **chromadb**: 向量数据库（当前未使用，可跳过）
- **langchain**: LangChain 框架（当前未使用，可跳过）
- **bitsandbytes**: 模型量化（仅 GPU 需要）
- **accelerate**: 模型加速库

### 文档处理依赖

- **PyPDF2**: PDF 文件处理
- **python-docx**: Word 文档处理
- **openpyxl**: Excel 文件处理（当前未使用）

## 系统要求

- **Python**: 3.8 或更高版本
- **内存**: 建议 8GB 以上
- **磁盘**: 至少 5GB 可用空间（用于模型缓存）
- **GPU**: 可选，用于加速推理

## 安装时间估算

- **基础安装**: 5-10 分钟
- **包含模型下载**: 10-30 分钟（取决于网络）

## 卸载依赖

如果需要卸载：

```bash
pip uninstall -r requirements.txt -y
```

## 更新依赖

```bash
pip install -r requirements.txt --upgrade
```

