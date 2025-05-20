import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
import os

# Set your OpenAI API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "api_key"

model=OllamaLLM(model="gemma3")

st.title("AI Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
# 1. Load and split documents
if uploaded_file is not None:
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # 1. Load the document
    loader = PyPDFLoader("temp_uploaded_file.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # 2. Embed and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 3. Create retriever and QA chain
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    rag_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, return_source_documents=True)


    query = st.text_input("Ask a question:")
    if st.button("Ask"):
        with st.spinner("retrieving data....."):
            result = rag_chain(query)
            st.write(result["result"])