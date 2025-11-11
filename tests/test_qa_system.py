"""
知识库问答系统测试脚本
"""
import os
import sys
from pathlib import Path
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kb_qa import KnowledgeBaseQA


def create_sample_documents():
    """创建示例文档用于测试"""
    docs_dir = Path("sample_documents")
    docs_dir.mkdir(exist_ok=True)
    
    # 创建示例文档1
    doc1_path = docs_dir / "人工智能介绍.txt"
    doc1_content = """人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
旨在创建能够执行通常需要人类智能的任务的系统。

主要技术包括：
1. 机器学习：让计算机从数据中学习
2. 深度学习：使用神经网络进行复杂模式识别
3. 自然语言处理：让计算机理解和生成人类语言
4. 计算机视觉：让计算机理解和分析图像

应用领域：
- 自动驾驶汽车
- 医疗诊断
- 推荐系统
- 语音助手
- 图像识别

人工智能的发展历史可以追溯到20世纪50年代，当时科学家们开始探索机器是否能够思考。
"""
    doc1_path.write_text(doc1_content, encoding='utf-8')
    
    # 创建示例文档2
    doc2_path = docs_dir / "机器学习基础.txt"
    doc2_content = """机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。

主要类型：
1. 监督学习：使用标记数据训练模型
   - 分类：预测离散类别
   - 回归：预测连续值

2. 无监督学习：从未标记数据中发现模式
   - 聚类：将相似数据分组
   - 降维：减少数据维度

3. 强化学习：通过与环境交互学习最优策略

常用算法：
- 线性回归
- 决策树
- 随机森林
- 支持向量机
- 神经网络

评估指标：
- 准确率
- 精确率
- 召回率
- F1分数
"""
    doc2_path.write_text(doc2_content, encoding='utf-8')
    
    # 创建示例文档3
    doc3_path = docs_dir / "深度学习.txt"
    doc3_content = """深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。

核心概念：
1. 神经网络：由多个层组成的计算模型
2. 反向传播：训练神经网络的关键算法
3. 激活函数：引入非线性，如ReLU、Sigmoid、Tanh

常见架构：
- 卷积神经网络（CNN）：用于图像处理
- 循环神经网络（RNN）：用于序列数据
- 长短期记忆网络（LSTM）：改进的RNN
- Transformer：用于自然语言处理

应用：
- 图像分类和识别
- 语音识别
- 机器翻译
- 文本生成
- 游戏AI（如AlphaGo）

训练过程：
1. 前向传播：数据通过网络
2. 计算损失：比较预测和真实值
3. 反向传播：更新权重
4. 重复直到收敛
"""
    doc3_path.write_text(doc3_content, encoding='utf-8')
    
    print(f"示例文档已创建在: {docs_dir}")
    return str(docs_dir)


def test_without_llm():
    """测试不使用LLM的版本（仅检索）"""
    print("=" * 60)
    print("测试1: 不使用LLM的检索模式")
    print("=" * 60)
    
    # 创建示例文档
    docs_dir = create_sample_documents()
    
    # 初始化系统（不加载LLM）
    print("\n初始化系统（模拟模式）...")
    qa_system = KnowledgeBaseQA(
        llama_model_name="dummy",  # 使用不存在的模型名，触发模拟模式
        vector_store_path="test_index"
    )
    
    # 构建知识库
    print("\n构建知识库...")
    qa_system.build_knowledge_base(docs_dir, chunk_size=300, chunk_overlap=50)
    
    # 测试查询
    test_questions = [
        "什么是人工智能？",
        "机器学习有哪些类型？",
        "深度学习的核心概念是什么？",
        "神经网络如何训练？"
    ]
    
    print("\n" + "=" * 60)
    print("开始测试查询")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 60)
        result = qa_system.query(question, top_k=2)
        print(f"答案: {result['answer']}")
        print(f"\n检索到的文档数量: {len(result['sources'])}")
        if result['sources']:
            print(f"最相关文档: {result['sources'][0]['content'][:100]}...")
        print()


def test_with_llm():
    """测试使用LLM的完整版本"""
    print("=" * 60)
    print("测试2: 使用LLM的完整问答系统")
    print("=" * 60)
    print("\n注意: 此测试需要下载LLaMA模型，可能需要较长时间")
    print("如果模型未下载，将自动切换到模拟模式\n")
    
    # 创建示例文档
    docs_dir = create_sample_documents()
    
    # 初始化系统
    print("\n初始化系统...")
    try:
        # 尝试使用较小的模型或本地模型
        # 如果使用 HuggingFace 模型，需要先登录: huggingface-cli login
        qa_system = KnowledgeBaseQA(
            llama_model_name="meta-llama/Llama-2-7b-chat-hf",  # 需要HuggingFace访问权限
            use_quantization=True,  # 使用量化以节省内存
            vector_store_path="test_index_llm"
        )
    except Exception as e:
        print(f"LLM模型加载失败: {e}")
        print("切换到模拟模式...")
        qa_system = KnowledgeBaseQA(
            llama_model_name="dummy",
            vector_store_path="test_index_llm"
        )
    
    # 构建知识库
    print("\n构建知识库...")
    qa_system.build_knowledge_base(docs_dir, chunk_size=300, chunk_overlap=50)
    
    # 测试查询
    test_questions = [
        "请详细解释什么是人工智能？",
        "监督学习和无监督学习有什么区别？",
        "深度学习的训练过程是怎样的？"
    ]
    
    print("\n" + "=" * 60)
    print("开始测试查询")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 60)
        result = qa_system.query(question, top_k=2, max_length=256)
        print(f"答案:\n{result['answer']}")
        print(f"\n检索到的文档数量: {len(result['sources'])}")
        print()


def main():
    """主测试函数"""
    print("=" * 60)
    print("知识库问答系统测试")
    print("=" * 60)
    
    # 测试1: 不使用LLM（快速测试）
    test_without_llm()
    
    # 询问是否测试LLM版本
    print("\n" + "=" * 60)
    response = input("是否测试使用LLM的完整版本？(需要下载模型，可能需要较长时间) [y/N]: ")
    if response.lower() == 'y':
        test_with_llm()
    else:
        print("跳过LLM测试。如需测试，请运行: python test_qa_system.py")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

