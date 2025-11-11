"""
简化测试脚本 - 快速验证系统功能
"""
from pathlib import Path
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kb_qa import KnowledgeBaseQA


def main():
    print("=" * 60)
    print("知识库问答系统 - 快速测试")
    print("=" * 60)
    
    # 创建示例文档
    print("\n1. 创建示例文档...")
    docs_dir = Path("sample_documents")
    docs_dir.mkdir(exist_ok=True)
    
    # 创建测试文档
    test_doc = docs_dir / "测试文档.txt"
    test_content = """Python是一种高级编程语言，由Guido van Rossum在1991年首次发布。

Python的特点：
- 语法简洁清晰
- 易于学习和使用
- 跨平台支持
- 丰富的标准库和第三方库

主要应用领域：
1. Web开发：使用Django、Flask等框架
2. 数据科学：NumPy、Pandas、Matplotlib
3. 人工智能：TensorFlow、PyTorch
4. 自动化脚本：系统管理、任务自动化

Python的版本：
- Python 2.x（已停止维护）
- Python 3.x（当前主流版本）

安装Python可以通过官方网站下载，或使用包管理器如apt、brew等。
"""
    test_doc.write_text(test_content, encoding='utf-8')
    print(f"✓ 示例文档已创建: {test_doc}")
    
    # 初始化系统（不使用LLM，仅测试检索功能）
    print("\n2. 初始化系统（检索模式）...")
    qa_system = KnowledgeBaseQA(
        llama_model_name="dummy",  # 触发模拟模式
        vector_store_path="test_simple_index"
    )
    print("✓ 系统初始化完成")
    
    # 构建知识库
    print("\n3. 构建知识库...")
    qa_system.build_knowledge_base(str(docs_dir), chunk_size=200, chunk_overlap=30)
    print("✓ 知识库构建完成")
    
    # 显示统计信息
    stats = qa_system.get_stats()
    print(f"\n知识库统计: {stats}")
    
    # 测试查询
    print("\n4. 测试查询功能...")
    print("=" * 60)
    
    questions = [
        "Python是什么？",
        "Python有哪些应用领域？",
        "如何安装Python？"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 60)
        result = qa_system.query(question, top_k=2)
        print(f"答案:\n{result['answer']}")
        print(f"\n检索到 {len(result['sources'])} 个相关文档片段")
        if result['sources']:
            print(f"最相关片段: {result['sources'][0]['content'][:150]}...")
    
    print("\n" + "=" * 60)
    print("✓ 测试完成！")
    print("=" * 60)
    print("\n提示:")
    print("- 当前使用的是检索模式（不使用LLM生成）")
    print("- 要使用完整的LLM问答功能，请配置LLaMA模型")
    print("- 运行 python test_qa_system.py 进行完整测试")


if __name__ == "__main__":
    main()

