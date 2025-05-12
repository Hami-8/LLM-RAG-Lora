"""
merge_lora.py
-------------
把 LoRA 微调结果合并到基模型，导出为独立权重
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==== 配置路径 ====
BASE_MODEL_DIR   = "/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct"      # 基座模型
LORA_CHECKPOINT  = "./output/llama3_1_instruct_lora/checkpoint-699"   # LoRA checkpoint 目录
MERGED_SAVE_DIR  = "/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct_Lora"                    # 导出目录

def main():
    # 1. 加载基模型（可用 4bit/8bit 节省显存，也可纯 CPU）
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        device_map="auto",            # 单GPU: "cuda:0"; CPU: None
        torch_dtype=torch.float16,    # bfloat16 亦可
    )

    # 2. 载入 LoRA 权重形成 PeftModel
    lora_model = PeftModel.from_pretrained(base, LORA_CHECKPOINT)

    # 3. 合并 LoRA → 普通模型，并卸掉 LoRA adapter
    merged_model = lora_model.merge_and_unload()   

    # 4. 保存合并后的权重
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True)
    merged_model.save_pretrained(MERGED_SAVE_DIR, safe_serialization=True)  # .safetensors
    tokenizer.save_pretrained(MERGED_SAVE_DIR)

    print(f"已保存到: {MERGED_SAVE_DIR}")

if __name__ == "__main__":
    main()
