#!/bin/bash
# 依赖安装脚本

echo "=========================================="
echo "知识库问答系统 - 依赖安装"
echo "=========================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

echo "Python 版本:"
python3 --version
echo ""

# 检查 pip
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "错误: 未找到 pip"
    exit 1
fi

# 升级 pip
echo "升级 pip..."
python3 -m pip install --upgrade pip --quiet
echo "✓ pip 已升级"
echo ""

# 询问安装方式
echo "选择安装方式:"
echo "1. 标准安装（推荐）"
echo "2. 使用国内镜像（清华源）"
echo "3. 使用国内镜像（阿里云源）"
echo "4. 最小化安装（仅核心功能）"
echo ""
read -p "请选择 [1-4] (默认: 1): " choice
choice=${choice:-1}

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_DIR"

case $choice in
    1)
        echo ""
        echo "开始标准安装..."
        pip3 install -r requirements.txt
        ;;
    2)
        echo ""
        echo "使用清华镜像安装..."
        pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        ;;
    3)
        echo ""
        echo "使用阿里云镜像安装..."
        pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
        ;;
    4)
        echo ""
        echo "开始最小化安装（仅核心功能）..."
        pip3 install torch transformers sentence-transformers faiss-cpu numpy pandas PyPDF2 python-docx python-dotenv
        ;;
    *)
        echo "无效选择，使用标准安装"
        pip3 install -r requirements.txt
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 依赖安装完成！"
    echo "=========================================="
    echo ""
    echo "验证安装..."
    python3 -c "
import sys
try:
    import torch
    import transformers
    import sentence_transformers
    import faiss
    print('✓ 所有核心依赖安装成功！')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  Transformers: {transformers.__version__}')
except ImportError as e:
    print(f'✗ 导入失败: {e}')
    sys.exit(1)
" && echo "" && echo "可以开始使用系统了！运行: python3 tests/simple_test.py"
else
    echo ""
    echo "=========================================="
    echo "✗ 安装失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi

