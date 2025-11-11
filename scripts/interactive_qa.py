"""
交互式知识库问答系统
提供命令行交互界面
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kb_qa import KnowledgeBaseQA


class InteractiveQA:
    """交互式问答界面"""
    
    def __init__(self):
        self.qa_system = None
        self.knowledge_base_built = False
        
    def print_banner(self):
        """打印欢迎信息"""
        print("=" * 60)
        print("基于 LLaMA 的个人知识库问答系统")
        print("=" * 60)
        print("\n可用命令:")
        print("  build <文档路径>  - 构建知识库")
        print("  stats            - 查看知识库统计")
        print("  help             - 显示帮助信息")
        print("  quit/exit        - 退出系统")
        print("\n直接输入问题即可开始问答")
        print("=" * 60 + "\n")
    
    def print_help(self):
        """显示帮助信息"""
        print("\n" + "=" * 60)
        print("帮助信息")
        print("=" * 60)
        print("""
命令说明:
  build <路径>    构建知识库
                  示例: build ./documents
                  示例: build document.txt
  
  stats           显示知识库统计信息
  
  help            显示此帮助信息
  
  quit/exit       退出系统

使用说明:
  1. 首先使用 'build' 命令构建知识库
  2. 然后直接输入问题开始问答
  3. 系统会自动检索相关文档并生成答案

示例:
  > build sample_documents
  > Python是什么？
  > 机器学习有哪些类型？
        """)
        print("=" * 60 + "\n")
    
    def build_knowledge_base(self, path: str):
        """构建知识库"""
        path = path.strip()
        if not path:
            print("错误: 请提供文档路径")
            return
        
        doc_path = Path(path)
        if not doc_path.exists():
            print(f"错误: 路径不存在: {path}")
            return
        
        print(f"\n正在构建知识库: {path}")
        print("这可能需要一些时间，请稍候...\n")
        
        try:
            # 初始化系统（如果还未初始化）
            if self.qa_system is None:
                print("初始化系统...")
                self.qa_system = KnowledgeBaseQA(
                    vector_store_path='interactive_index'
                )
            
            # 构建知识库
            self.qa_system.build_knowledge_base(
                str(doc_path),
                chunk_size=500,
                chunk_overlap=50
            )
            
            self.knowledge_base_built = True
            stats = self.qa_system.get_stats()
            print(f"\n✓ 知识库构建完成！")
            print(f"  文档数量: {stats['total_documents']}")
            print(f"  向量维度: {stats['dimension']}")
            print()
            
        except Exception as e:
            print(f"\n✗ 构建知识库失败: {e}\n")
    
    def show_stats(self):
        """显示统计信息"""
        if not self.knowledge_base_built or self.qa_system is None:
            print("错误: 请先构建知识库 (使用 'build' 命令)")
            return
        
        stats = self.qa_system.get_stats()
        print("\n" + "=" * 60)
        print("知识库统计信息")
        print("=" * 60)
        print(f"总文档数: {stats['total_documents']}")
        print(f"索引大小: {stats['index_size']}")
        print(f"向量维度: {stats['dimension']}")
        print("=" * 60 + "\n")
    
    def query(self, question: str):
        """查询"""
        if not self.knowledge_base_built or self.qa_system is None:
            print("错误: 请先构建知识库 (使用 'build' 命令)")
            return
        
        if not question.strip():
            return
        
        print("\n" + "-" * 60)
        print(f"问题: {question}")
        print("-" * 60)
        
        try:
            result = self.qa_system.query(question, top_k=3, max_length=512)
            
            print(f"\n答案:\n{result['answer']}")
            
            if result['sources']:
                print(f"\n参考文档 ({len(result['sources'])} 个):")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"  [{i}] {source['metadata'].get('file_name', '未知')}")
                    print(f"      相似度: {1/(1+source['distance']):.2%}")
            
            print("\n" + "-" * 60 + "\n")
            
        except Exception as e:
            print(f"\n✗ 查询失败: {e}\n")
    
    def run(self):
        """运行交互式界面"""
        self.print_banner()
        
        while True:
            try:
                # 获取用户输入
                user_input = input("> ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n再见！")
                    break
                
                elif user_input.lower() == 'help':
                    self.print_help()
                
                elif user_input.lower() == 'stats':
                    self.show_stats()
                
                elif user_input.lower().startswith('build '):
                    path = user_input[6:].strip()
                    self.build_knowledge_base(path)
                
                else:
                    # 作为问题处理
                    self.query(user_input)
            
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except EOFError:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {e}\n")


def main():
    """主函数"""
    app = InteractiveQA()
    app.run()


if __name__ == "__main__":
    main()

