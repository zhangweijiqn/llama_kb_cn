"""
基于 LLaMA 的个人知识库问答系统
整合文档加载、向量检索和 RAG 生成
"""
import os
from typing import List, Dict, Optional
from pathlib import Path

from .document_loader import DocumentLoader
from .vector_store import VectorStore
from .llama_model import LLaMAModel


class KnowledgeBaseQA:
    """知识库问答系统"""
    
    def __init__(self, 
                 embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 llama_model_name: str = "Qwen/Qwen2-0.5B-Instruct",  # 更小的中文模型，约500MB
                 use_quantization: bool = False,
                 vector_store_path: Optional[str] = None,
                 only_knowledge_base: bool = True,
                 similarity_threshold: float = 0.0):
        """
        初始化知识库问答系统
        
        Args:
            embedding_model: 嵌入模型名称
            llama_model_name: LLaMA 模型名称或路径
            use_quantization: 是否使用量化
            vector_store_path: 向量存储路径
            only_knowledge_base: 是否仅在知识库内回答（默认 True）
                                - True: 仅基于知识库内容回答
                                - False: 可以回答知识库之外的问题
            similarity_threshold: 相似度阈值 (0.0-1.0)，用于过滤低相关性结果
                                - 0.0: 不过滤（默认，返回所有结果）
                                - 0.5: 中等严格（过滤相似度 < 50% 的结果）
                                - 0.7: 较严格（过滤相似度 < 70% 的结果）
                                - 0.8: 很严格（只返回相似度 >= 80% 的结果）
        """
        print("初始化知识库问答系统...")
        
        # 保存配置
        self.only_knowledge_base = only_knowledge_base
        self.similarity_threshold = similarity_threshold
        
        # 初始化文档加载器
        self.doc_loader = DocumentLoader()
        
        # 初始化向量存储
        self.vector_store = VectorStore(
            embedding_model_name=embedding_model,
            index_path=vector_store_path or 'knowledge_base_index',
            data_dir='data'  # 数据文件保存到 data 目录
        )
        
        # 初始化 LLaMA 模型（添加安全处理）
        self.llm = None
        if llama_model_name and llama_model_name.lower() != 'dummy':
            try:
                # 检查是否强制跳过模型加载
                skip_model = os.environ.get('SKIP_LLM_MODEL', 'false').lower() == 'true'
                if skip_model:
                    print("⚠ 环境变量 SKIP_LLM_MODEL=true，跳过模型加载")
                    print("将使用模拟模式（仅返回检索结果）")
                else:
                    self.llm = LLaMAModel(
                        model_name=llama_model_name,
                        use_quantization=use_quantization
                    )
            except (RuntimeError, OSError, SystemError) as e:
                error_str = str(e).lower()
                if 'segmentation' in error_str or 'memory' in error_str or 'killed' in error_str:
                    print(f"\n⚠ 模型加载失败（可能是内存问题）: {type(e).__name__}")
                    print("将使用模拟模式（仅返回检索结果）")
                    print("\n建议解决方案:")
                    print("  1. 使用量化: use_quantization=True")
                    print("  2. 使用更小的模型")
                    print("  3. 设置环境变量: export SKIP_LLM_MODEL=true")
                    print("  4. 增加系统内存")
                else:
                    print(f"⚠ 模型加载失败: {e}")
                    print("将使用模拟模式（仅返回检索结果）")
                self.llm = None
            except Exception as e:
                print(f"⚠ 模型加载失败: {e}")
                print("将使用模拟模式（仅返回检索结果）")
                self.llm = None
        else:
            print("使用模拟模式（仅返回检索结果）")
        
        print("系统初始化完成")
    
    def build_knowledge_base(self, document_path: str, chunk_size: int = 500, 
                            chunk_overlap: int = 50):
        """
        构建知识库
        
        Args:
            document_path: 文档路径（文件或目录）
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        print(f"开始构建知识库: {document_path}")
        
        document_path = Path(document_path)
        
        # 加载文档
        if document_path.is_file():
            documents = [self.doc_loader.load_document(str(document_path))]
        elif document_path.is_dir():
            documents = self.doc_loader.load_directory(str(document_path))
        else:
            raise ValueError(f"路径不存在: {document_path}")
        
        print(f"加载了 {len(documents)} 个文档")
        
        # 分割文档并添加到向量存储
        all_chunks = []
        all_metadatas = []
        
        for doc in documents:
            chunks = self.doc_loader.split_text(
                doc['content'],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({
                    'file_name': doc['file_name'],
                    'file_path': doc['file_path'],
                    'file_type': doc['file_type']
                })
        
        print(f"生成了 {len(all_chunks)} 个文本块")
        
        # 添加到向量存储
        self.vector_store.add_documents(all_chunks, all_metadatas)
        
        # 保存向量存储
        self.vector_store.save()
        
        print("知识库构建完成")
        print(f"统计信息: {self.vector_store.get_stats()}")
    
    def query(self, question: str, top_k: int = 3, max_length: int = 512, max_new_tokens: int = None,
              temperature: float = 0.7, similarity_threshold: Optional[float] = None) -> Dict[str, any]:
        """
        查询知识库
        
        Args:
            question: 问题
            top_k: 检索的文档数量
            max_length: 最大总长度（已弃用，建议使用 max_new_tokens）
            max_new_tokens: 最大新生成的 token 数量（推荐使用，默认 256）
            temperature: 生成温度
            similarity_threshold: 相似度阈值，如果为 None 则使用初始化时的阈值
            
        Returns:
            包含答案和相关文档的字典
        """
        print(f"查询: {question}")
        
        # 使用传入的阈值或默认阈值
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        
        # 1. 检索相关文档
        search_results = self.vector_store.search(question, k=top_k, similarity_threshold=threshold)
        
        # 打印检索结果信息（用于调试）
        if search_results:
            for i, (text, distance, metadata) in enumerate(search_results, 1):
                similarity = 1.0 / (1.0 + distance)
                print(f"  检索结果 {i}: 相似度={similarity:.2%}, 距离={distance:.4f}")
        else:
            print(f"  ⚠ 未找到相似度 >= {threshold:.2%} 的结果")
        
        # 2. 检查是否仅在知识库内回答
        if self.only_knowledge_base:
            # 仅在知识库内回答模式
            if not search_results:
                return {
                    'answer': '抱歉，知识库中没有找到相关信息。请确保问题与知识库内容相关。',
                    'sources': [],
                    'retrieved_docs': []
                }
            
            # 构建上下文（仅使用知识库内容）
            context_parts = []
            retrieved_docs = []
            
            for i, (text, distance, metadata) in enumerate(search_results, 1):
                context_parts.append(f"[文档{i}]\n{text}\n")
                retrieved_docs.append({
                    'rank': i,
                    'content': text[:200] + '...' if len(text) > 200 else text,
                    'distance': float(distance),
                    'metadata': metadata
                })
            
            context = '\n'.join(context_parts)
            
            # 构建提示（严格限制在知识库内）
            if self.llm:
                prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请明确说明"根据提供的文档，无法回答此问题"。

文档内容：
{context}

问题：{question}

请基于上述文档内容，用中文回答问题。如果文档中没有相关信息，请明确说明："""
        else:
            # 允许回答知识库外的问题
            context_parts = []
            retrieved_docs = []
            
            if search_results:
                # 如果有检索结果，使用它们作为参考
                for i, (text, distance, metadata) in enumerate(search_results, 1):
                    context_parts.append(f"[文档{i}]\n{text}\n")
                    retrieved_docs.append({
                        'rank': i,
                        'content': text[:200] + '...' if len(text) > 200 else text,
                        'distance': float(distance),
                        'metadata': metadata
                    })
                context = '\n'.join(context_parts)
                context_note = f"\n\n参考文档内容：\n{context}\n\n如果文档中没有相关信息，你可以基于自己的知识回答。"
            else:
                # 没有检索结果，直接让模型基于自己的知识回答
                context = ""
                context_note = "\n\n注意：知识库中没有找到相关信息，请基于你的知识回答这个问题。"
            
            # 构建提示（允许知识库外回答）
            if self.llm:
                prompt = f"""请回答以下问题。{context_note}

问题：{question}

请用中文回答问题："""
        
        # 4. 生成答案
        if self.llm:
            try:
                # 如果没有指定 max_new_tokens，使用默认值 256
                if max_new_tokens is None:
                    max_new_tokens = 256
                
                answer = self.llm.generate(
                    prompt,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                # 检查答案是否为空
                if not answer or len(answer.strip()) == 0:
                    print("⚠ 警告: 生成的答案为空")
                    # 如果答案为空，根据模式处理
                    if self.only_knowledge_base:
                        # 仅在知识库模式：使用检索结果
                        if search_results:
                            answer = f"根据知识库检索，相关信息如下：\n\n{search_results[0][0][:500]}"
                        else:
                            answer = "抱歉，知识库中没有相关信息。"
                    else:
                        # 允许知识库外模式：提示模型回答
                        if not search_results:
                            # 没有检索结果，尝试让模型直接回答
                            direct_prompt = f"请回答以下问题：\n\n{question}\n\n请用中文回答问题："
                            try:
                                answer = self.llm.generate(
                                    direct_prompt,
                                    max_length=max_length,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature
                                )
                                if not answer or len(answer.strip()) == 0:
                                    answer = "抱歉，无法生成答案。"
                            except:
                                answer = "抱歉，无法生成答案。"
                        else:
                            answer = f"根据知识库检索，相关信息如下：\n\n{search_results[0][0][:500]}"
                else:
                    # 确保答案不为空，如果太短且有检索结果，补充检索结果
                    if len(answer.strip()) < 20 and search_results:
                        answer += f"\n\n参考信息：\n{search_results[0][0][:300]}"
                    
            except Exception as e:
                print(f"生成答案时出错: {e}")
                import traceback
                traceback.print_exc()
                # 如果生成失败，根据模式处理
                if self.only_knowledge_base:
                    # 仅在知识库模式：使用检索结果
                    if search_results:
                        answer = f"根据知识库检索，最相关的信息如下：\n\n{search_results[0][0][:500]}"
                    else:
                        answer = "抱歉，知识库中没有相关信息。"
                else:
                    # 允许知识库外模式：尝试让模型直接回答
                    if not search_results:
                        direct_prompt = f"请回答以下问题：\n\n{question}\n\n请用中文回答问题："
                        try:
                            answer = self.llm.generate(
                                direct_prompt,
                                max_length=max_length,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature
                            )
                            if not answer or len(answer.strip()) == 0:
                                answer = "抱歉，无法生成答案。"
                        except:
                            answer = "抱歉，无法生成答案。"
                    else:
                        answer = f"根据知识库检索，最相关的信息如下：\n\n{search_results[0][0][:500]}"
        else:
            # 模拟模式：只返回最相关的文档片段
            if search_results:
                answer = f"根据知识库检索，最相关的信息如下：\n\n{search_results[0][0][:500]}"
            else:
                if self.only_knowledge_base:
                    answer = "抱歉，知识库中没有相关信息。"
                else:
                    answer = "抱歉，模型未加载，无法回答知识库外的问题。"
        
        return {
            'answer': answer,
            'sources': retrieved_docs,
            'retrieved_docs': retrieved_docs
        }
    
    def chat(self, question: str, conversation_history: Optional[List[dict]] = None,
             top_k: int = 3, max_length: int = 512, max_new_tokens: int = None,
             temperature: float = 0.7, similarity_threshold: Optional[float] = None) -> Dict[str, any]:
        """
        对话式查询（支持上下文）
        
        Args:
            question: 当前问题
            conversation_history: 对话历史
            top_k: 检索的文档数量
            max_length: 最大总长度（已弃用，建议使用 max_new_tokens）
            max_new_tokens: 最大新生成的 token 数量（推荐使用，默认 256）
            temperature: 生成温度
            similarity_threshold: 相似度阈值，如果为 None 则使用初始化时的阈值
            
        Returns:
            包含答案和相关文档的字典
        """
        # 使用传入的阈值或默认阈值
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        
        # 检索相关文档
        search_results = self.vector_store.search(question, k=top_k, similarity_threshold=threshold)
        
        # 构建上下文
        context_parts = []
        retrieved_docs = []
        
        for i, (text, distance, metadata) in enumerate(search_results, 1):
            context_parts.append(f"[文档{i}]\n{text}\n")
            retrieved_docs.append({
                'rank': i,
                'content': text[:200] + '...' if len(text) > 200 else text,
                'distance': float(distance),
                'metadata': metadata
            })
        
        context = '\n'.join(context_parts) if context_parts else ""
        
        # 构建消息
        messages = []
        
        # 添加系统提示
        if self.only_knowledge_base:
            # 仅在知识库内回答
            if not search_results:
                return {
                    'answer': '抱歉，知识库中没有找到相关信息。请确保问题与知识库内容相关。',
                    'sources': [],
                    'retrieved_docs': []
                }
            system_content = f"你是一个基于知识库的智能助手。请严格基于以下文档内容回答问题。如果文档中没有相关信息，请明确说明\"根据提供的文档，无法回答此问题\"。\n\n文档内容：\n{context}"
        else:
            # 允许回答知识库外的问题
            if search_results:
                system_content = f"你是一个智能助手。请优先基于以下文档内容回答问题。如果文档中没有相关信息，可以基于你的知识回答。\n\n参考文档内容：\n{context}"
            else:
                system_content = "你是一个智能助手。知识库中没有找到相关信息，请基于你的知识回答这个问题。"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # 添加对话历史
        if conversation_history:
            messages.extend(conversation_history)
        
        # 添加当前问题
        messages.append({
            "role": "user",
            "content": question
        })
        
        # 生成答案
        if self.llm:
            try:
                # 如果没有指定 max_new_tokens，使用默认值 256
                if max_new_tokens is None:
                    max_new_tokens = 256
                
                answer = self.llm.chat(messages, max_length=max_length, max_new_tokens=max_new_tokens, temperature=temperature)
                
                # 检查答案是否为空
                if not answer or len(answer.strip()) == 0:
                    print("⚠ 警告: 生成的答案为空")
                    # 根据模式处理
                    if self.only_knowledge_base:
                        # 仅在知识库模式
                        if search_results:
                            answer = f"根据知识库检索，相关信息如下：\n\n{search_results[0][0][:500]}"
                        else:
                            answer = "抱歉，无法生成答案。"
                    else:
                        # 允许知识库外模式
                        if not search_results:
                            # 没有检索结果，尝试让模型直接回答
                            direct_messages = [
                                {"role": "system", "content": "你是一个智能助手。请用中文回答问题。"},
                                {"role": "user", "content": question}
                            ]
                            try:
                                answer = self.llm.chat(
                                    direct_messages,
                                    max_length=max_length,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature
                                )
                                if not answer or len(answer.strip()) == 0:
                                    answer = "抱歉，无法生成答案。"
                            except:
                                answer = "抱歉，无法生成答案。"
                        else:
                            answer = f"根据知识库检索，相关信息如下：\n\n{search_results[0][0][:500]}"
                else:
                    # 确保答案不为空，如果太短且有检索结果，补充检索结果
                    if len(answer.strip()) < 20 and search_results:
                        answer += f"\n\n参考信息：\n{search_results[0][0][:300]}"
                        
            except Exception as e:
                print(f"生成答案时出错: {e}")
                import traceback
                traceback.print_exc()
                # 如果生成失败，根据模式处理
                if self.only_knowledge_base:
                    # 仅在知识库模式
                    if search_results:
                        answer = f"根据知识库检索，最相关的信息如下：\n\n{search_results[0][0][:500]}"
                    else:
                        answer = "抱歉，无法生成答案。"
                else:
                    # 允许知识库外模式
                    if not search_results:
                        # 没有检索结果，尝试让模型直接回答
                        direct_messages = [
                            {"role": "system", "content": "你是一个智能助手。请用中文回答问题。"},
                            {"role": "user", "content": question}
                        ]
                        try:
                            answer = self.llm.chat(
                                direct_messages,
                                max_length=max_length,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature
                            )
                            if not answer or len(answer.strip()) == 0:
                                answer = "抱歉，无法生成答案。"
                        except:
                            answer = "抱歉，无法生成答案。"
                    else:
                        answer = f"根据知识库检索，最相关的信息如下：\n\n{search_results[0][0][:500]}"
        else:
            # 模拟模式：只返回最相关的文档片段
            if search_results:
                answer = f"根据知识库检索，最相关的信息如下：\n\n{search_results[0][0][:500]}"
            else:
                if self.only_knowledge_base:
                    answer = "抱歉，知识库中没有相关信息。"
                else:
                    answer = "抱歉，模型未加载，无法回答知识库外的问题。"
        
        return {
            'answer': answer,
            'sources': retrieved_docs,
            'retrieved_docs': retrieved_docs
        }
    
    def clear_knowledge_base(self):
        """清空知识库"""
        self.vector_store.clear()
        print("✓ 知识库已清空")
    
    def has_content(self) -> bool:
        """检查知识库是否有内容"""
        return not self.vector_store.is_empty()
    
    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        return self.vector_store.get_stats()

