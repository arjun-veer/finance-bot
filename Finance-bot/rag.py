import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_file in pdf_docs:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")

def retrieve_documents(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question, k=5)  # Limit to top 5 relevant chunks
    return docs

def main():
    st.set_page_config(page_title="Finance Chatbot", layout="wide")
    st.header("Finance Chatbot 💬")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Upload and Process PDFs")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully!")
            else:
                st.error("Please upload at least one PDF file.")

    # Chat interface
    st.subheader("Ask Your Financial Questions")
    user_question = st.text_input("Type your question here:")
    if st.button("Ask"):
        if user_question.strip():
            with st.spinner("Retrieving answer..."):
                retrieved_docs = retrieve_documents(user_question)
                if retrieved_docs:
                    # Combine retrieved chunks into a single context
                    context = " ".join([doc.page_content for doc in retrieved_docs])
                    st.write("**Relevant Information:**", context)
                else:
                    st.write("No relevant information found.")
        else:
            st.error("Please enter a question.")

if __name__ == "__main__":
    main()
