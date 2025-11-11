#!/bin/bash
# 启动 Web 应用脚本

echo "=========================================="
echo "知识库问答系统 - Web 界面"
echo "=========================================="
echo ""

# 检查是否安装了 streamlit
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "正在安装 streamlit..."
    pip install streamlit
fi

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_DIR"

echo "启动 Web 应用..."
echo "浏览器将自动打开，如果没有，请访问: http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

# 启动 Streamlit
streamlit run web_app.py

