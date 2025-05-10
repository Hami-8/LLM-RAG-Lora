# rag_local.py
from langchain_core.globals import set_verbose, set_debug
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

import torch, logging, os

set_debug(True)
set_verbose(True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatPDF:
    """
    RAG over local PDF with a locally‑downloaded LLM & embedding model.
    """

    def __init__(
        self,
        llm_path: str = "/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct",          # 本地 LLM 目录
        embed_path: str = "/root/autodl-tmp/mirror013/mxbai-embed-large-v1",   # 本地向量模型目录
        gpu: bool = True,
    ):
        # ------- 1) LLM --------
        logger.info(f"Loading LLM from {llm_path}")
        tok = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if gpu else None,
        )
        gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tok.eos_token_id,   # 保证遇到 </s> 就停
            return_full_text=False
        )
        self.model = HuggingFacePipeline(pipeline=gen_pipeline)

        # ------- 2) Embeddings --------
        logger.info(f"Loading embeddings from {embed_path}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_path,
            model_kwargs={"device": "cuda" if gpu else "cpu"},
        )

        # ------- 3) 其他组件 --------
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.
            Context:
            {context}

            Question:
            {question}

            Answer concisely and accurately in three sentences or less.
            """
        )
        self.vector_store = None
        self.retriever = None

    # ---------- 与原版保持一致的接口 ----------
    def ingest(self, pdf_file_path: str):
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db",
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            # 只取 Top‑k，不做阈值过滤
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": k}
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)
        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        chain = (
            RunnablePassthrough() | self.prompt | self.model | StrOutputParser()
        )
        logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)

    def clear(self):
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None


# ====== 简单用法示例 ======
if __name__ == "__main__":
    rag = ChatPDF(
        llm_path="./local_models/Meta-Llama-3-1.8B-Instruct-merged",
        embed_path="./local_models/all-MiniLM-L6-v2",
    )
    rag.ingest("demo.pdf")
    print(rag.ask("本文档的研究动机是什么？"))
