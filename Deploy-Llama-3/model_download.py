import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
model_dir_2 = snapshot_download('mirror013/mxbai-embed-large-v1', cache_dir='/root/autodl-tmp', revision='master')

# model_dir_3 = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', cache_dir='/root/autodl-tmp', revision='master')