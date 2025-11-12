"""
LLaMA 模型加载和推理模块
支持使用 transformers 库加载 LLaMA 模型
"""
import os
import sys
import torch
from typing import List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

# 设置环境变量以避免某些问题
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


class LLaMAModel:
    """LLM 模型包装类（支持 LLaMA、Qwen 等模型）"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct", 
                 use_quantization: bool = True,
                 device: Optional[str] = None):
        """
        初始化 LLM 模型
        
        Args:
            model_name: 模型名称或路径
                - 默认模型: "Qwen/Qwen2-0.5B-Instruct" (约500MB，最小)
                - 其他选项: "Qwen/Qwen2-1.5B-Instruct" (约1.8GB), "Qwen/Qwen-1.8B-Chat"
            use_quantization: 是否使用量化（8bit）以节省内存
            device: 设备（'cuda' 或 'cpu'），None 表示自动选择
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"正在加载模型: {model_name}")
        print(f"使用设备: {self.device}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 配置量化（如果需要）
        model_kwargs = {}
        if use_quantization and self.device == 'cuda':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            model_kwargs['quantization_config'] = quantization_config
            model_kwargs['device_map'] = 'auto'
        
        # 加载模型（添加更多安全措施）
        try:
            # 设置较低的内存使用
            if self.device == 'cpu':
                # CPU 模式使用 float32，避免内存问题
                model_kwargs.setdefault('dtype', torch.float32)
                # 限制线程数
                torch.set_num_threads(2)
            else:
                model_kwargs.setdefault('dtype', torch.float16)
            
            # 使用低 CPU 内存模式
            if self.device == 'cpu' and not use_quantization:
                model_kwargs['low_cpu_mem_usage'] = True
            
            print("正在下载/加载模型，这可能需要几分钟...")
            print("如果遇到内存问题，请尝试：")
            print("  1. 使用 use_quantization=True")
            print("  2. 使用更小的模型")
            print("  3. 增加系统内存或使用 GPU")
            print("  4. 设置环境变量: export SKIP_LLM_MODEL=true")
            
            # 添加更多安全参数，避免 segfault
            safe_kwargs = {
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,  # 降低 CPU 内存使用
                'dtype': model_kwargs.get('dtype', torch.float32),
            }
            
            # 合并用户指定的参数
            safe_kwargs.update(model_kwargs)
            
            # 如果是 CPU 模式，添加更多限制
            if self.device == 'cpu':
                # 限制线程数，避免内存问题
                torch.set_num_threads(1)
                # 使用更保守的设置
                safe_kwargs['low_cpu_mem_usage'] = True
                # 避免使用 device_map（可能导致问题）
                if 'device_map' in safe_kwargs:
                    del safe_kwargs['device_map']
            
            # 分步加载，避免一次性加载导致 segfault
            try:
                # 先尝试加载配置，检查模型是否可用
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                print(f"模型配置加载成功: {config.model_type}")
                
                # 加载模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config,
                    **safe_kwargs
                )
            except (RuntimeError, OSError, SystemError) as e:
                error_str = str(e).lower()
                if 'segmentation' in error_str or 'killed' in error_str or 'signal' in error_str:
                    print(f"\n✗ Segmentation fault 或内存错误")
                    print("\n这通常是由于内存不足导致的。")
                    print("\n建议解决方案:")
                    print("  1. 设置环境变量跳过模型: export SKIP_LLM_MODEL=true")
                    print("  2. 使用更小的模型或量化版本")
                    print("  3. 增加系统内存")
                    print("  4. 使用 GPU（如果有）")
                    raise RuntimeError("模型加载失败：内存不足或系统限制") from e
                raise
            
            # 移动到设备
            if not use_quantization:
                try:
                    self.model.to(self.device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"内存不足错误: {e}")
                        print("建议: 使用 use_quantization=True 或更小的模型")
                        raise
                    raise
            
            self.model.eval()
            print("✓ 模型加载完成")
            
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "segmentation" in error_msg.lower() or "killed" in error_msg.lower():
                print(f"\n✗ 内存错误或 Segmentation Fault: {error_msg}")
                print("\n解决方案:")
                print("  1. 设置环境变量跳过模型: export SKIP_LLM_MODEL=true")
                print("  2. 使用量化版本: use_quantization=True")
                print("  3. 使用更小的模型或跳过模型加载")
                print("  4. 关闭其他程序释放内存")
                print("  5. 使用 GPU（如果有）")
                print("\n提示: 系统会自动降级到检索模式（不使用 LLM）")
            raise
        except (OSError, SystemError) as e:
            error_msg = str(e).lower()
            if 'segmentation' in error_msg or 'signal' in error_msg or 'killed' in error_msg:
                print(f"\n✗ Segmentation Fault 错误")
                print("\n这通常是由于:")
                print("  1. 内存不足")
                print("  2. 系统资源限制")
                print("  3. 依赖库版本不兼容")
                print("\n建议:")
                print("  1. 设置环境变量: export SKIP_LLM_MODEL=true")
                print("  2. 更新依赖: pip install --upgrade torch transformers")
                print("  3. 使用检索模式（不使用 LLM）")
                raise RuntimeError("模型加载失败：Segmentation Fault") from e
            raise
        except Exception as e:
            print(f"\n✗ 加载模型失败: {e}")
            print("\n可能的原因:")
            print("  1. 网络问题：模型下载失败")
            print("  2. 权限问题：需要 HuggingFace 登录")
            print("  3. 依赖问题：transformers 版本不兼容")
            print("\n解决方案:")
            print("  1. 检查网络连接")
            print("  2. 运行: huggingface-cli login")
            print("  3. 更新: pip install --upgrade transformers torch")
            print("  4. 使用本地模型路径")
            raise
    
    def generate(self, prompt: str, max_length: int = 512, max_new_tokens: int = None,
                 temperature: float = 0.7, top_p: float = 0.9,
                 do_sample: bool = True) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大总长度（输入+输出），已弃用，建议使用 max_new_tokens
            max_new_tokens: 最大新生成的 token 数量（推荐使用）
            temperature: 温度参数（控制随机性）
            top_p: nucleus sampling 参数
            do_sample: 是否使用采样
            
        Returns:
            生成的文本
        """
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        
        # 使用 max_new_tokens 而不是 max_length，避免输入长度问题
        generate_kwargs = {
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': do_sample,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'repetition_penalty': 1.1
        }
        
        if max_new_tokens is not None:
            # 优先使用 max_new_tokens（推荐）
            generate_kwargs['max_new_tokens'] = max_new_tokens
        else:
            # 如果没有指定 max_new_tokens，计算合适的值
            # 确保总长度不超过 max_length，但至少生成一些内容
            if input_length >= max_length:
                # 输入已经很长，只生成少量新 token
                generate_kwargs['max_new_tokens'] = min(256, max_length - input_length + 50)
                print(f"⚠ 输入长度 ({input_length}) 接近或超过 max_length ({max_length})，使用 max_new_tokens={generate_kwargs['max_new_tokens']}")
            else:
                # 使用 max_length，但建议使用 max_new_tokens
                generate_kwargs['max_length'] = max_length
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs
            )
        
        # 解码输出
        full_generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 获取输入 token 数量（用于提取新生成的部分）
        input_length = inputs['input_ids'].shape[1]
        output_length = outputs[0].shape[0]
        
        # 方法1: 优先使用 token 长度提取（最可靠）
        if output_length > input_length:
            # 只解码新生成的部分（从 input_length 开始）
            new_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        else:
            # 方法2: 如果输出长度等于输入长度，说明没有生成新内容
            generated_text = ""
        
        # 方法3: 如果方法1失败，尝试字符串匹配
        if not generated_text or len(generated_text.strip()) == 0:
            # 尝试从完整文本中移除 prompt
            if prompt in full_generated_text:
                generated_text = full_generated_text[len(prompt):].strip()
            elif len(full_generated_text) > input_length * 2:  # 如果文本明显比输入长
                # 尝试移除前 input_length 个字符（粗略估计）
                generated_text = full_generated_text[input_length:].strip()
        
        # 如果仍然为空，尝试其他方法
        if not generated_text or len(generated_text.strip()) == 0:
            print("⚠ 警告: 模型生成的文本为空或提取失败")
            print(f"  输入长度: {input_length}, 输出长度: {output_length}")
            print(f"  完整文本长度: {len(full_generated_text)}")
            print(f"  Prompt 长度: {len(prompt)}")
            
            # 如果完整文本比 prompt 长，至少返回一些内容
            if len(full_generated_text) > len(prompt) + 10:
                generated_text = full_generated_text[len(prompt):].strip()
            else:
                # 最后尝试：返回完整文本（可能包含 prompt，但至少有内容）
                generated_text = full_generated_text.strip()
                if generated_text == prompt:
                    generated_text = "抱歉，模型未能生成有效答案。请查看参考来源获取相关信息。"
        
        # 清理文本：移除可能的截断标记和不自然的结尾
        # 检查是否在句子中间被截断（以常见的中文标点结尾）
        if generated_text:
            # 移除末尾的截断标记（如 "..." 或 "…"）
            generated_text = generated_text.rstrip('...…')
            
            # 如果文本不以句号、问号、感叹号结尾，且长度较长，可能是被截断了
            # 但不要强制添加，让用户看到完整的生成内容
        
        return generated_text
    
    def chat(self, messages: List[dict], max_length: int = 512, max_new_tokens: int = None,
             temperature: float = 0.7) -> str:
        """
        对话格式生成（适用于 chat 模型）
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
            max_length: 最大总长度（已弃用，建议使用 max_new_tokens）
            max_new_tokens: 最大新生成的 token 数量（推荐使用）
            temperature: 温度参数
            
        Returns:
            生成的回复
        """
        # 构建提示
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # 如果 chat_template 不可用，使用简单格式
                prompt = self._build_simple_prompt(messages)
        else:
            # 简单的提示格式
            prompt = self._build_simple_prompt(messages)
        
        return self.generate(prompt, max_length=max_length, max_new_tokens=max_new_tokens, temperature=temperature)
    
    def _build_simple_prompt(self, messages: List[dict]) -> str:
        """构建简单的提示格式"""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"系统: {content}\n\n"
            elif role == "user":
                prompt += f"用户: {content}\n\n"
            elif role == "assistant":
                prompt += f"助手: {content}\n\n"
        prompt += "助手: "
        return prompt


class LLaMALiteModel:
    """
    轻量级 LLaMA 模型（使用 llama.cpp）
    如果 transformers 版本加载失败，可以使用这个替代方案
    """
    
    def __init__(self, model_path: str):
        """
        初始化轻量级模型
        
        Args:
            model_path: 模型文件路径（.gguf 格式）
        """
        try:
            from llama_cpp import Llama
            self.llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
            print("轻量级模型加载完成")
        except ImportError:
            raise ImportError("需要安装 llama-cpp-python: pip install llama-cpp-python")
    
    def generate(self, prompt: str, max_tokens: int = 256, 
                 temperature: float = 0.7) -> str:
        """生成文本"""
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\n"],
            echo=False
        )
        return response['choices'][0]['text']

