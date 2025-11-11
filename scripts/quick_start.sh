#!/bin/bash
# 快速启动脚本

echo "=========================================="
echo "知识库问答系统 - 快速启动"
echo "=========================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import sentence_transformers, faiss, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "正在安装依赖..."
    pip install sentence-transformers faiss-cpu torch transformers --quiet
fi

echo ""
echo "选择启动方式:"
echo "1. 交互式界面 (推荐)"
echo "2. 快速测试"
echo "3. 完整测试"
echo ""
read -p "请选择 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "启动交互式界面..."
        cd "$(dirname "$0")"
        python3 interactive_qa.py
        ;;
    2)
        echo ""
        echo "运行快速测试..."
        cd "$(dirname "$0")/../tests"
        python3 simple_test.py
        ;;
    3)
        echo ""
        echo "运行完整测试..."
        cd "$(dirname "$0")/../tests"
        python3 test_qa_system.py
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

