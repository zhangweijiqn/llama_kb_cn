#!/usr/bin/env python
"""安全测试脚本 - 逐步测试模型加载"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_step_by_step():
    """逐步测试模型加载"""
    print("=" * 60)
    print("安全模型加载测试")
    print("=" * 60)
    print()
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # 步骤1: 测试导入
    print("步骤1: 测试依赖导入...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ Transformers 导入成功")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    print()
    
    # 步骤2: 测试配置加载
    print("步骤2: 测试模型配置加载...")
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ 配置加载成功: {config.model_type}")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False
    print()
    
    # 步骤3: 测试 tokenizer 加载
    print("步骤3: 测试 Tokenizer 加载...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Tokenizer 加载成功")
    except Exception as e:
        print(f"✗ Tokenizer 加载失败: {e}")
        return False
    print()
    
    # 步骤4: 测试模型加载（使用最安全的设置）
    print("步骤4: 测试模型加载（安全模式）...")
    print("  使用最保守的设置...")
    try:
        # 最安全的设置
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        print("✓ 模型加载成功（float32 模式）")
        del model  # 释放内存
        import gc
        gc.collect()
    except Exception as e:
        print(f"✗ 模型加载失败: {type(e).__name__}: {e}")
        print()
        print("建议:")
        print("  1. 设置环境变量: export SKIP_LLM_MODEL=true")
        print("  2. 使用检索模式（不使用 LLM）")
        return False
    print()
    
    # 步骤5: 测试完整系统
    print("步骤5: 测试完整系统（跳过模型）...")
    try:
        from kb_qa import KnowledgeBaseQA
        qa = KnowledgeBaseQA(llama_model_name="dummy")
        print("✓ 系统初始化成功（检索模式）")
    except Exception as e:
        print(f"✗ 系统初始化失败: {e}")
        return False
    print()
    
    print("=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    print()
    print("如果模型加载失败，可以使用检索模式:")
    print("  qa = KnowledgeBaseQA(llama_model_name='dummy')")
    print("  或设置: export SKIP_LLM_MODEL=true")
    
    return True

if __name__ == "__main__":
    success = test_step_by_step()
    sys.exit(0 if success else 1)

