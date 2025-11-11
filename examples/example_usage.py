"""
知识库问答系统使用示例
展示如何使用系统进行问答
"""
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kb_qa import KnowledgeBaseQA
from pathlib import Path


def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("示例1: 基础使用")
    print("=" * 60)
    
    # 初始化系统
    qa_system = KnowledgeBaseQA(
        embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
        llama_model_name="dummy",  # 使用模拟模式
        vector_store_path='example_index'
    )
    
    # 如果有文档目录，构建知识库
    docs_dir = Path("sample_documents")
    if docs_dir.exists():
        qa_system.build_knowledge_base(str(docs_dir))
    
    # 查询
    question = "Python是什么？"
    result = qa_system.query(question, top_k=2)
    
    print(f"\n问题: {question}")
    print(f"答案: {result['answer']}")
    print(f"来源文档数: {len(result['sources'])}")


def example_with_custom_documents():
    """使用自定义文档的示例"""
    print("\n" + "=" * 60)
    print("示例2: 使用自定义文档")
    print("=" * 60)
    
    # 创建自定义文档
    custom_doc = Path("custom_doc.txt")
    custom_doc.write_text("""
机器学习是人工智能的核心技术之一。

监督学习使用标记数据训练模型，包括：
- 分类任务：预测离散类别
- 回归任务：预测连续数值

无监督学习从未标记数据中发现模式，包括：
- 聚类：将相似数据分组
- 降维：减少数据维度

强化学习通过与环境交互学习最优策略。
""", encoding='utf-8')
    
    # 初始化并构建知识库
    qa_system = KnowledgeBaseQA(
        vector_store_path='custom_index'
    )
    qa_system.build_knowledge_base(str(custom_doc), chunk_size=200)
    
    # 查询
    questions = [
        "什么是监督学习？",
        "无监督学习包括哪些任务？"
    ]
    
    for q in questions:
        result = qa_system.query(q)
        print(f"\n问题: {q}")
        print(f"答案: {result['answer'][:200]}...")


def example_conversation():
    """对话示例"""
    print("\n" + "=" * 60)
    print("示例3: 对话模式")
    print("=" * 60)
    
    qa_system = KnowledgeBaseQA(vector_store_path='conversation_index')
    
    # 构建知识库
    docs_dir = Path("sample_documents")
    if docs_dir.exists():
        qa_system.build_knowledge_base(str(docs_dir))
    
    # 对话历史
    history = []
    
    # 第一轮对话
    question1 = "Python是什么？"
    result1 = qa_system.chat(question1, conversation_history=history)
    print(f"\n用户: {question1}")
    print(f"助手: {result1['answer'][:150]}...")
    
    # 更新历史
    history.append({"role": "user", "content": question1})
    history.append({"role": "assistant", "content": result1['answer']})
    
    # 第二轮对话（带上下文）
    question2 = "它有哪些特点？"
    result2 = qa_system.chat(question2, conversation_history=history)
    print(f"\n用户: {question2}")
    print(f"助手: {result2['answer'][:150]}...")


def example_with_llm():
    """使用LLM的完整示例（需要配置模型）"""
    print("\n" + "=" * 60)
    print("示例4: 使用LLM生成答案")
    print("=" * 60)
    print("注意: 此示例需要配置LLaMA模型")
    print("如果模型未配置，将自动使用检索模式\n")
    
    try:
        # 尝试使用LLM（需要HuggingFace访问权限或本地模型）
        qa_system = KnowledgeBaseQA(
            llama_model_name="meta-llama/Llama-2-7b-chat-hf",
            use_quantization=True,  # 使用量化节省内存
            vector_store_path='llm_index'
        )
        print("✓ LLaMA模型加载成功")
    except Exception as e:
        print(f"⚠ LLaMA模型加载失败: {e}")
        print("使用检索模式...")
        qa_system = KnowledgeBaseQA(
            llama_model_name="dummy",
            vector_store_path='llm_index'
        )
    
    # 构建知识库
    docs_dir = Path("sample_documents")
    if docs_dir.exists():
        qa_system.build_knowledge_base(str(docs_dir))
    
    # 查询（如果LLM可用，会生成更完整的答案）
    question = "请详细解释Python的特点和应用"
    result = qa_system.query(question, top_k=3, max_length=256)
    
    print(f"\n问题: {question}")
    print(f"答案:\n{result['answer']}")
    print(f"\n参考了 {len(result['sources'])} 个文档片段")


def main():
    """运行所有示例"""
    print("\n知识库问答系统 - 使用示例\n")
    
    # 运行示例
    example_basic_usage()
    example_with_custom_documents()
    example_conversation()
    example_with_llm()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

