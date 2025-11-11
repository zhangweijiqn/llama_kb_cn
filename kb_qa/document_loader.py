"""
文档加载和预处理模块
支持多种文档格式：PDF、Word、TXT、Markdown
"""
import os
import re
from typing import List, Dict
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None


class DocumentLoader:
    """文档加载器，支持多种格式"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.md', '.pdf', '.docx']
    
    def load_document(self, file_path: str) -> Dict[str, any]:
        """
        加载文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            包含文档内容和元数据的字典
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt' or suffix == '.md':
            return self._load_text(file_path)
        elif suffix == '.pdf':
            return self._load_pdf(file_path)
        elif suffix == '.docx':
            return self._load_docx(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def _load_text(self, file_path: Path) -> Dict[str, any]:
        """加载文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            'content': content,
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': file_path.suffix
        }
    
    def _load_pdf(self, file_path: Path) -> Dict[str, any]:
        """加载PDF文件"""
        if PyPDF2 is None:
            raise ImportError("需要安装 PyPDF2: pip install PyPDF2")
        
        content = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                content.append(page.extract_text())
        
        return {
            'content': '\n'.join(content),
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': '.pdf'
        }
    
    def _load_docx(self, file_path: Path) -> Dict[str, any]:
        """加载Word文档"""
        if Document is None:
            raise ImportError("需要安装 python-docx: pip install python-docx")
        
        doc = Document(file_path)
        content = []
        for paragraph in doc.paragraphs:
            content.append(paragraph.text)
        
        return {
            'content': '\n'.join(content),
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': '.docx'
        }
    
    def split_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        将文本分割成块
        
        Args:
            text: 要分割的文本
            chunk_size: 每个块的大小（字符数）
            chunk_overlap: 块之间的重叠大小
            
        Returns:
            文本块列表
        """
        # 先按段落分割
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            # 如果单个段落就超过chunk_size，需要进一步分割
            if para_length > chunk_size:
                # 先保存当前chunk
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # 按句子分割长段落
                sentences = re.split(r'[。！？\n]', para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if current_length + len(sentence) > chunk_size:
                        if current_chunk:
                            chunks.append('\n'.join(current_chunk))
                            # 保留重叠部分
                            overlap_text = '\n'.join(current_chunk[-chunk_overlap//50:])
                            current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                            current_length = len('\n'.join(current_chunk))
                        else:
                            current_chunk = [sentence]
                            current_length = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_length += len(sentence) + 1
            else:
                # 检查添加这个段落是否会超过chunk_size
                if current_length + para_length > chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    # 保留重叠部分
                    overlap_text = '\n'.join(current_chunk[-chunk_overlap//50:])
                    current_chunk = [overlap_text, para] if overlap_text else [para]
                    current_length = len('\n'.join(current_chunk))
                else:
                    current_chunk.append(para)
                    current_length += para_length + 1
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def load_directory(self, directory: str) -> List[Dict[str, any]]:
        """
        加载目录中的所有支持格式的文档
        
        Args:
            directory: 目录路径
            
        Returns:
            文档列表
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        documents = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self.load_document(file_path)
                    documents.append(doc)
                except Exception as e:
                    print(f"加载文件失败 {file_path}: {e}")
        
        return documents

