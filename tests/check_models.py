#!/usr/bin/env python
"""检查已下载的模型"""

import os
import sys

def get_model_cache_dir():
    """获取模型缓存目录"""
    try:
        from transformers import file_utils
        return file_utils.default_cache_path
    except:
        # 备用方法
        home = os.path.expanduser("~")
        if sys.platform == "darwin":  # macOS
            return os.path.join(home, ".cache", "huggingface")
        elif sys.platform == "win32":  # Windows
            return os.path.join(home, ".cache", "huggingface")
        else:  # Linux
            return os.path.join(home, ".cache", "huggingface")

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def get_dir_size(path):
    """计算目录大小"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except:
        pass
    return total

def main():
    print("=" * 60)
    print("HuggingFace 模型缓存检查")
    print("=" * 60)
    print()
    
    cache_dir = get_model_cache_dir()
    print(f"缓存目录: {cache_dir}")
    print()
    
    if not os.path.exists(cache_dir):
        print("✗ 缓存目录不存在（尚未下载任何模型）")
        print()
        print("首次使用时会自动下载模型到此目录")
        return
    
    print("✓ 缓存目录存在")
    print()
    
    # 检查 hub 目录
    hub_dir = os.path.join(cache_dir, "hub")
    if not os.path.exists(hub_dir):
        print("✗ hub 目录不存在")
        return
    
    # 查找模型
    models = []
    for item in os.listdir(hub_dir):
        if item.startswith("models--"):
            model_path = os.path.join(hub_dir, item)
            if os.path.isdir(model_path):
                model_name = item.replace("models--", "").replace("--", "/")
                size = get_dir_size(model_path)
                models.append((model_name, model_path, size))
    
    if not models:
        print("(暂无已下载的模型)")
        print()
        print("首次使用时会自动下载模型")
        return
    
    print(f"找到 {len(models)} 个已下载的模型:")
    print("-" * 60)
    print()
    
    total_size = 0
    for model_name, model_path, size in sorted(models):
        total_size += size
        print(f"模型: {model_name}")
        print(f"  路径: {model_path}")
        print(f"  大小: {format_size(size)}")
        print()
    
    print("-" * 60)
    print(f"总大小: {format_size(total_size)}")
    print()
    
    # 显示如何设置自定义缓存目录
    print("=" * 60)
    print("如何设置自定义缓存目录")
    print("=" * 60)
    print()
    print("方法1: 设置环境变量")
    print("  export HF_HOME=/path/to/your/cache")
    print()
    print("方法2: 在代码中指定")
    print("  from transformers import AutoModel")
    print("  model = AutoModel.from_pretrained(")
    print("      'Qwen/Qwen2-0.5B-Instruct',")
    print("      cache_dir='/path/to/your/cache'")
    print("  )")
    print()
    print("方法3: 查看环境变量")
    print("  echo $HF_HOME")
    print("  echo $TRANSFORMERS_CACHE")
    print()

if __name__ == "__main__":
    main()


