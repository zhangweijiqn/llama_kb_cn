#!/usr/bin/env python
"""
独立的模型加载测试脚本
用于测试模型是否能正常加载，不依赖其他组件
"""

import os
import sys
import traceback

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_imports():
    """测试依赖导入"""
    print_section("步骤1: 测试依赖导入")
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        import transformers
        print(f"✓ Transformers 版本: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers 导入失败: {e}")
        return False
    
    return True


def test_model_config(model_name):
    """测试模型配置加载"""
    print_section(f"步骤2: 测试模型配置加载 ({model_name})")
    
    try:
        from transformers import AutoConfig
        
        print(f"正在加载配置...")
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        print(f"✓ 配置加载成功")
        print(f"  模型类型: {config.model_type}")
        print(f"  隐藏层大小: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"  层数: {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"  注意力头数: {getattr(config, 'num_attention_heads', 'N/A')}")
        
        return True, config
    except Exception as e:
        print(f"✗ 配置加载失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None


def test_tokenizer(model_name):
    """测试 Tokenizer 加载"""
    print_section(f"步骤3: 测试 Tokenizer 加载 ({model_name})")
    
    try:
        from transformers import AutoTokenizer
        
        print(f"正在加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        print(f"✓ Tokenizer 加载成功")
        print(f"  词汇表大小: {len(tokenizer)}")
        print(f"  Pad token: {tokenizer.pad_token}")
        print(f"  EOS token: {tokenizer.eos_token}")
        
        # 测试编码
        test_text = "你好，这是一个测试。"
        encoded = tokenizer(test_text, return_tensors="pt")
        print(f"  测试编码: '{test_text}' -> {encoded['input_ids'].shape}")
        
        return True, tokenizer
    except Exception as e:
        print(f"✗ Tokenizer 加载失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None


def test_model_load_safe(model_name, use_quantization=False):
    """测试模型加载（安全模式）"""
    print_section(f"步骤4: 测试模型加载 ({model_name})")
    
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"量化模式: {use_quantization}")
    
    try:
        # 准备加载参数
        model_kwargs = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
        }
        
        if device == 'cpu':
            model_kwargs['torch_dtype'] = torch.float32
            torch.set_num_threads(1)  # 限制线程数
        else:
            model_kwargs['torch_dtype'] = torch.float16
        
        if use_quantization and device == 'cuda':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            model_kwargs['quantization_config'] = quantization_config
            model_kwargs['device_map'] = 'auto'
        
        print(f"正在加载模型（这可能需要几分钟）...")
        print(f"参数: {model_kwargs}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if not use_quantization:
            model = model.to(device)
        
        model.eval()
        
        print(f"✓ 模型加载成功")
        print(f"  模型类型: {type(model).__name__}")
        print(f"  设备: {next(model.parameters()).device}")
        print(f"  数据类型: {next(model.parameters()).dtype}")
        
        # 计算模型大小
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {param_count / 1e6:.2f}M")
        
        return True, model
    except RuntimeError as e:
        error_msg = str(e).lower()
        if 'out of memory' in error_msg or 'segmentation' in error_msg:
            print(f"\n✗ 内存错误或 Segmentation Fault")
            print(f"  错误: {e}")
            print(f"\n建议:")
            print(f"  1. 使用量化: use_quantization=True")
            print(f"  2. 使用更小的模型")
            print(f"  3. 增加系统内存")
        else:
            print(f"✗ 模型加载失败: {e}")
        traceback.print_exc()
        return False, None
    except (OSError, SystemError) as e:
        error_msg = str(e).lower()
        if 'segmentation' in error_msg or 'signal' in error_msg:
            print(f"\n✗ Segmentation Fault")
            print(f"  错误: {e}")
            print(f"\n这通常是由于内存不足导致的")
            print(f"建议: 使用 SKIP_LLM_MODEL=true 跳过模型加载")
        else:
            print(f"✗ 模型加载失败: {e}")
        traceback.print_exc()
        return False, None
    except Exception as e:
        print(f"✗ 模型加载失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None


def test_model_inference(model, tokenizer, device):
    """测试模型推理"""
    print_section("步骤5: 测试模型推理")
    
    try:
        import torch
        
        test_prompt = "你好，请介绍一下你自己。"
        print(f"测试提示: {test_prompt}")
        
        # 编码
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"正在生成（这可能需要几秒钟）...")
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除输入部分
        if test_prompt in generated_text:
            generated_text = generated_text[len(test_prompt):].strip()
        
        print(f"✓ 生成成功")
        print(f"生成结果: {generated_text}")
        
        return True
    except Exception as e:
        print(f"✗ 推理失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("模型加载测试脚本")
    print("=" * 60)
    print()
    print("此脚本用于测试模型是否能正常加载和运行")
    print("不依赖知识库系统的其他组件")
    print()
    
    # 默认模型
    default_model = "Qwen/Qwen2-0.5B-Instruct"
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='测试模型加载')
    parser.add_argument(
        '--model',
        type=str,
        default=default_model,
        help=f'模型名称 (默认: {default_model})'
    )
    parser.add_argument(
        '--quantization',
        action='store_true',
        help='使用量化模式（8-bit）'
    )
    parser.add_argument(
        '--skip-inference',
        action='store_true',
        help='跳过推理测试'
    )
    
    args = parser.parse_args()
    model_name = args.model
    use_quantization = args.quantization
    
    print(f"测试模型: {model_name}")
    print(f"量化模式: {use_quantization}")
    print()
    
    # 步骤1: 测试导入
    if not test_imports():
        print("\n✗ 依赖导入失败，请检查安装")
        return 1
    
    # 步骤2: 测试配置
    success, config = test_model_config(model_name)
    if not success:
        print("\n✗ 配置加载失败")
        return 1
    
    # 步骤3: 测试 tokenizer
    success, tokenizer = test_tokenizer(model_name)
    if not success:
        print("\n✗ Tokenizer 加载失败")
        return 1
    
    # 步骤4: 测试模型加载
    success, model = test_model_load_safe(model_name, use_quantization)
    if not success:
        print("\n✗ 模型加载失败")
        print("\n建议:")
        print("  1. 尝试使用量化: --quantization")
        print("  2. 使用更小的模型")
        print("  3. 设置环境变量: export SKIP_LLM_MODEL=true")
        return 1
    
    # 步骤5: 测试推理（可选）
    if not args.skip_inference:
        device = next(model.parameters()).device
        success = test_model_inference(model, tokenizer, device)
        if not success:
            print("\n⚠ 推理测试失败，但模型加载成功")
    
    # 清理
    import gc
    import torch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print_section("测试完成")
    print("✓ 所有测试通过！")
    print()
    print("模型可以正常使用。")
    print("如果遇到问题，可以:")
    print("  1. 使用量化模式: --quantization")
    print("  2. 跳过推理测试: --skip-inference")
    print("  3. 使用更小的模型")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n未预期的错误: {e}")
        traceback.print_exc()
        sys.exit(1)

