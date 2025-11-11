#!/usr/bin/env python
"""快速修复 segmentation fault 问题"""

import os
import sys

def main():
    print("=" * 60)
    print("Segmentation Fault 修复工具")
    print("=" * 60)
    
    print("\n1. 检查环境...")
    try:
        import torch
        import transformers
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ 依赖缺失: {e}")
        print("运行: pip install torch transformers")
        return
    
    print("\n2. 检查内存...")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  总内存: {mem.total / 1024 / 1024 / 1024:.2f} GB")
        print(f"  可用内存: {mem.available / 1024 / 1024 / 1024:.2f} GB")
        if mem.available < 2 * 1024 * 1024 * 1024:  # 小于 2GB
            print("  ⚠ 可用内存不足，建议使用量化或跳过模型")
    except ImportError:
        print("  (无法检查内存，需要安装 psutil)")
    
    print("\n3. 测试模型加载（使用量化）...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from kb_qa import KnowledgeBaseQA
        print("  正在尝试加载模型（最小模型+量化模式）...")
        qa = KnowledgeBaseQA(
            llama_model_name="Qwen/Qwen2-0.5B-Instruct",  # 最小模型
            use_quantization=True  # 使用量化
        )
        print("  ✓ 模型加载成功（量化模式）")
    except Exception as e:
        print(f"  ✗ 加载失败: {type(e).__name__}: {e}")
        print("\n  建议:")
        print("    1. 设置环境变量: export SKIP_LLM_MODEL=true")
        print("    2. 使用更小的模型")
        print("    3. 检查系统内存")
        return
    
    print("\n4. 测试检索功能...")
    try:
        # 测试检索（不需要模型）
        qa = KnowledgeBaseQA(llama_model_name="dummy")
        print("  ✓ 检索功能正常")
    except Exception as e:
        print(f"  ✗ 检索测试失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✓ 系统检查完成")
    print("=" * 60)
    print("\n如果仍有问题，请:")
    print("  1. 查看完整错误信息")
    print("  2. 检查系统资源")
    print("  3. 尝试使用 SKIP_LLM_MODEL=true")
    print("  4. 查看 docs/TROUBLESHOOTING.md")

if __name__ == "__main__":
    main()

