import os
import streamlit as st
import time
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Explicitly configure Gemini API

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini"

main_placeholder = st.empty()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    try:
        data = loader.load()
        if not data or all(doc.page_content.strip() == "" for doc in data):
            st.error("No valid content could be loaded from the provided URLs. Please check the URLs.")
        else:
            st.write(f"Loaded data from URLs: {[doc.page_content[:100] for doc in data]}")

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            if not docs:
                st.error("No valid text chunks could be generated. Please check the content of the URLs.")
            else:
                st.write(f"Generated {len(docs)} text chunks.")

                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Use Gemini embeddings
                    vectorstore_gemini = FAISS.from_documents(docs, embeddings)
                    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                    st.write(f"FAISS index contains {vectorstore_gemini.index.ntotal} vectors.")

                    vectorstore_gemini.save_local(file_path)
                    st.success("FAISS vector store created and saved successfully.")
                except Exception as e:
                    st.error(f"An error occurred during FAISS vector store creation: {e}")
    except Exception as e:
        st.error(f"An error occurred while loading data from URLs: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    try:
        if not os.path.exists(f"{file_path}/index.faiss") or not os.path.exists(f"{file_path}/index.pkl"):
            st.error("FAISS vector store files are missing or corrupted. Please process the URLs again.")
        else:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Use Gemini embeddings
            vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.3),  # Use Gemini API
                retriever=retriever
            )
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
    except Exception as e:
        st.error(f"An error occurred during query processing: {e}")