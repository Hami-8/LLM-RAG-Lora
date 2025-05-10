# Local ChatPDF with DeepSeek R1

**ChatPDF** is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them through a chatbot interface. The system uses advanced embedding models and a local vector store for efficient and accurate question-answering.

## Features

- **PDF Upload**: Upload one or multiple PDF documents to enable question-answering across their combined content.
- **RAG Workflow**: Combines retrieval and generation for high-quality responses.
- **Customizable Retrieval**: Adjust the number of retrieved results (`k`) and similarity threshold to fine-tune performance.
- **Memory Management**: Easily clear vector store and retrievers to reset the system.
- **Streamlit Interface**: A user-friendly web application for seamless interaction.

---

## Installation

Follow the steps below to set up and run the application:

### 1. Clone the Repository

```bash
git clone https://github.com/paquino11/chatpdf-rag-deepseek-r1.git
cd chatpdf-rag-deepseek-r1
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure to include the following packages in your `requirements.txt`:

```
streamlit
langchain
langchain_ollama
langchain_community
streamlit-chat
pypdf
chromadb
```

### 4. Pull Required Models for Ollama

To use the specified embedding and LLM models (`mxbai-embed-large` and `deepseek-r1`), download them via the `ollama` CLI:

```bash
ollama pull mxbai-embed-large
ollama pull deepseek-r1:latest
```

---

## Usage

### 1. Start the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

### 2. Upload Documents

- Navigate to the **Upload a Document** section in the web interface.
- Upload one or multiple PDF files to process their content.
- Each file will be ingested automatically and confirmation messages will show processing time.

### 3. Ask Questions

- Type your question in the chat input box and press Enter.
- Adjust retrieval settings (`k` and similarity threshold) in the **Settings** section for better responses.

### 4. Clear Chat and Reset

- Use the **Clear Chat** button to reset the chat interface.
- Clearing the chat also resets the vector store and retriever.

---

## Project Structure

```
.
├── app.py                  # Streamlit app for the user interface
├── rag.py                  # Core RAG logic for PDF ingestion and question-answering
├── requirements.txt        # List of required Python dependencies
├── chroma_db/              # Local persistent vector store (auto-generated)
└── README.md               # Project documentation
```

---

## Configuration

You can modify the following parameters in `rag.py` to suit your needs:

1. **Models**:
   - Default LLM: `deepseek-r1:latest` (7B parameters)
   - Default Embedding: `mxbai-embed-large` (1024 dimensions)
   - Change these in the `ChatPDF` class constructor or when initializing the class
   - Any Ollama-compatible model can be used by updating the `llm_model` parameter

2. **Chunking Parameters**:
   - `chunk_size=1024` and `chunk_overlap=100`
   - Adjust for larger or smaller document splits

3. **Retrieval Settings**:
   - Adjust `k` (number of retrieved results) and `score_threshold` in `ask()` to control the quality of retrieval.

---

## Requirements

- **Python**: 3.8+
- **Streamlit**: Web framework for the user interface.
- **Ollama**: For embedding and LLM models.
- **LangChain**: Core framework for RAG.
- **PyPDF**: For PDF document processing.
- **ChromaDB**: Vector store for document embeddings.

---

## Troubleshooting

### Common Issues

1. **Missing Models**:
   - Ensure you've pulled the required models using `ollama pull`.

2. **Vector Store Errors**:
   - Delete the `chroma_db/` directory if you encounter dimensionality errors:
     ```bash
     rm -rf chroma_db/
     ```

3. **Streamlit Not Launching**:
   - Verify dependencies are installed correctly using `pip install -r requirements.txt`.

---

## Future Enhancements

- **Memory Integration**: Add persistent memory to maintain conversational context across sessions.
- **Advanced Analytics**: Include visual insights from retrieved content.
- **Expanded Model Support**: Support additional embedding and LLM providers.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://github.com/streamlit/streamlit)
- [Ollama](https://ollama.ai/)

