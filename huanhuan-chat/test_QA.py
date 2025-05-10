from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# 模型与LoRA路径
mode_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora_path = './output/llama3_1_instruct_lora/checkpoint-10'

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    mode_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 初始 system message（你可以根据角色替换）
system_prompt = "假设你是皇帝身边的女人--甄嬛。"

# 开启交互式对话循环
print("🔮 已加载模型，现在你可以开始提问。输入 'exit' 退出。\n")

while True:
    user_input = input("👤 你：")
    if user_input.lower() in ["exit", "quit", "退出"]:
        print("👋 再见！")
        break

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # 构造 Prompt
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to('cuda')

    # 模型生成回复
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

    # 解码并输出回答
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]  # 去除 prompt 部分
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("嬛嬛：", response)
