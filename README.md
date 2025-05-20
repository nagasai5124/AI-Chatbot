# AI-Chatbot

AI-Chatbot is a simple Streamlit-based application that allows users to interact with PDF documents using a conversational AI interface. Upload a PDF file and ask questions about its content—answers are generated using Retrieval-Augmented Generation (RAG) with language models and document embeddings.

## Features

- Upload PDF documents and ask questions about their content.
- Uses LangChain and FAISS for document embedding and retrieval.
- Powered by HuggingFace embeddings and Ollama LLM (with the "gemma3" model).
- Streamlit web interface for easy interaction.

## How It Works

1. **Upload PDF**: Users upload a PDF file through the Streamlit interface.
2. **Document Processing**: The PDF is loaded and split into chunks for efficient retrieval.
3. **Embeddings & Vector Store**: Chunks are embedded using HuggingFace models and stored in a FAISS vector database.
4. **Question Answering**: User questions are processed using a RetrievalQA chain that searches the document and generates answers.

## Installation

Clone the repository:

```bash
git clone https://github.com/nagasai5124/AI-Chatbot.git
cd AI-Chatbot
```

Install the required dependencies:

```bash
pip install -r req.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

1. Upload your PDF file in the app.
2. Type your question in the input box and click "Ask".
3. View the AI-generated answer based on your document.

## Requirements

Dependencies listed in `req.txt`:

```
langchain
faiss-cpu
streamlit
langchain-core
langchain-community
langchain-ollama
pypdf
sentence-transformers
```

## Configuration

- Ensure you have your HuggingFace API key set in your environment or directly in the script if testing.
- Ollama LLM is used with the "gemma3" model.

## output screenshot and video
![Streamlit and 1 more page - Personal - Microsoft​ Edge 5_20_2025 4_23_27 PM](https://github.com/user-attachments/assets/ee929e55-4384-4487-add2-ea1b097fdf83)
![Streamlit and 1 more page - Personal - Microsoft​ Edge 5_20_2025 4_37_42 PM](https://github.com/user-attachments/assets/f283163f-897e-4c47-bc40-b2dc86bcdc96)
![Streamlit and 1 more page - Personal - Microsoft​ Edge 5_20_2025 4_40_52 PM](https://github.com/user-attachments/assets/f6708bfa-f161-42c7-85c7-88febb3b02e0)


