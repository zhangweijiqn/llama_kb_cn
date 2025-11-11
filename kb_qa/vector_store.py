"""
向量数据库模块
使用 FAISS 进行向量存储和检索
"""
import os
import pickle
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer


class VectorStore:
    """基于 FAISS 的向量存储"""
    
    def __init__(self, embedding_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', 
                 dimension: int = 384, index_path: Optional[str] = None, data_dir: str = 'data'):
        """
        初始化向量存储
        
        Args:
            embedding_model_name: 嵌入模型名称
            dimension: 向量维度
            index_path: 索引保存路径（文件名，不含扩展名）
            data_dir: 数据存储目录，默认为 'data'
        """
        if faiss is None:
            raise ImportError("需要安装 faiss-cpu: pip install faiss-cpu")
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = dimension
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)  # 确保目录存在
        
        # 构建完整路径
        index_name = index_path or 'vector_store'
        self.index_path = self.data_dir / index_name
        
        # 创建 FAISS 索引（使用 L2 距离）
        self.index = faiss.IndexFlatL2(dimension)
        
        # 存储文本和元数据
        self.texts = []
        self.metadata = []
        
        # 如果存在已保存的索引，加载它
        if (self.index_path.with_suffix('.index')).exists():
            self.load()
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """
        添加文档到向量存储
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表（可选）
        """
        if not texts:
            return
        
        # 生成嵌入向量
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # 添加到 FAISS 索引
        self.index.add(embeddings.astype('float32'))
        
        # 保存文本和元数据
        self.texts.extend(texts)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{}] * len(texts))
    
    def search(self, query: str, k: int = 5, similarity_threshold: float = 0.0) -> List[Tuple[str, float, dict]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            k: 返回最相似的 k 个结果
            similarity_threshold: 相似度阈值 (0.0-1.0)，低于此阈值的结果将被过滤
                                  - 0.0: 不过滤（默认）
                                  - 0.5: 中等严格
                                  - 0.7: 较严格
                                  - 0.8: 很严格
            
        Returns:
            (文本, 距离, 元数据) 的列表，已按相似度过滤
        """
        if self.index.ntotal == 0:
            return []
        
        # 生成查询向量
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # 搜索（获取更多结果以便过滤）
        search_k = min(k * 3, self.index.ntotal)  # 搜索更多结果以便过滤
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # 构建结果并应用阈值过滤
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.texts):
                # 将 L2 距离转换为相似度分数 (0-1)
                # 相似度 = 1 / (1 + distance)，距离越小，相似度越高
                similarity = 1.0 / (1.0 + float(distance))
                
                # 应用阈值过滤
                if similarity >= similarity_threshold:
                    results.append((
                        self.texts[idx],
                        float(distance),
                        self.metadata[idx]
                    ))
                    
                    # 如果已经收集到足够的符合阈值的结果，提前退出
                    if len(results) >= k:
                        break
        
        return results
    
    def save(self):
        """保存索引和元数据"""
        # 保存 FAISS 索引
        index_file = self.index_path.with_suffix('.index')
        faiss.write_index(self.index, str(index_file))
        
        # 保存文本和元数据
        pkl_file = self.index_path.with_suffix('.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)
    
    def load(self):
        """加载索引和元数据"""
        # 加载 FAISS 索引
        index_file = self.index_path.with_suffix('.index')
        self.index = faiss.read_index(str(index_file))
        
        # 加载文本和元数据
        pkl_file = self.index_path.with_suffix('.pkl')
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']
            self.dimension = data['dimension']
    
    def clear(self):
        """清空向量存储"""
        # 清空索引
        self.index.reset()
        
        # 清空文本和元数据
        self.texts = []
        self.metadata = []
        
        # 删除保存的文件
        index_file = self.index_path.with_suffix('.index')
        pkl_file = self.index_path.with_suffix('.pkl')
        
        if index_file.exists():
            index_file.unlink()
        if pkl_file.exists():
            pkl_file.unlink()
        
        print("✓ 向量存储已清空")
    
    def is_empty(self) -> bool:
        """检查向量存储是否为空"""
        return len(self.texts) == 0 and self.index.ntotal == 0
    
    def get_stats(self) -> dict:
        """获取向量存储统计信息"""
        return {
            'total_documents': len(self.texts),
            'index_size': self.index.ntotal,
            'dimension': self.dimension
        }

