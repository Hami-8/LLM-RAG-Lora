from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 原始模型路径
model_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

# 交互式问答
print("🤖 模型已加载，现在可以提问了。输入 'exit' 退出。")

system_prompt = "You are a helpful assistant answering questions"

while True:
    user_input = input("👤 你：")
    if user_input.lower() in ["exit", "quit", "退出"]:
        print("👋 再见！")
        break

    # 构造 prompt（LLaMA3 格式）
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to('cuda')

    # 模型生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # 解码输出
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("模型：", response)
