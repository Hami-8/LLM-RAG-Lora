# import torch
# from modelscope import snapshot_download, AutoModel, AutoTokenizer
# import os

# model_dir = snapshot_download('mxbai/embed-large', cache_dir='/root/autodl-tmp', revision='master')

# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("mxbai/embed-large")
# model.save("/root/autodl-tmp/mxbai-embed-large")

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('mirror013/mxbai-embed-large-v1', cache_dir='/root/autodl-tmp', revision='master')

