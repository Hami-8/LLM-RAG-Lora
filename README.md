# LLM-RAG-Lora

本项目基于 Meta-Llama-3.1-8B-Instruct 进行RAG和Lora微调。

## Clone the Repository

```
git clone https://github.com/Hami-8/LLM-RAG-Lora.git
cd LLM-RAG-Lora
```

## Enviroment

本项目基础环境如下：

```
----------------
python 3.12
cuda 12.1
pytorch 2.3.0
----------------
```

安装所依赖的包：

```
pip install -r requirements.txt
```

## 部署 Llama-3.1-8B

首先下载 Meta-Llama-3.1-8B-Instruct 模型和 mxbai-embed-large-v1 模型
- mxbai-embed-large-v1 模型用于构建RAG。

```
cd Deploy-Llama-3
python model_download.py
```

进行交互式问答测试

```
python test_QA_initial.py
```


## RAG

RAG项目参考 [该仓库](https://github.com/paquino11/chatpdf-rag-deepseek-r1) 进行改进。

通过streamlit打开网页进行交互，在网页上可上传PDF作为RAG的内容。
```
cd RAG-Llama-3
streamlit run app.py
```

网页如下图：

![alt text](image.png)

## Lora

Lora项目参考 [该仓库](https://github.com/KMnO4-zx/huanhuan-chat.git) 进行改进。

### 训练

```
cd Lora-Llama-3
python train.py
```

训练出的Lora参数会保存在 llm_rag_lora/Lora-Llama-3/output/llama3_1_instruct_lora 中。

### 测试

三种测试文件

- test_QA_initial.py 不附加Lora的交互式问答测试。
- test_QA.py 附加Lora的交互式问答测试。
- test.py 附加Lora的非交互式测试。


```
python test_QA.py
```
