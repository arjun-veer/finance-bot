import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# genai.configure(model="gemini-2.5-pro-exp-03-25")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_file in pdf_docs:  # Iterate over the list of uploaded files
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
    # return vector_store 

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, give small answer in general which are outof pdf also, if the answer is not related to finance then  just say, "ask question related to finance", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model=ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Allow deserialization
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    resposne = chain({
        "input_documents": docs,
        "question": user_question
    }, return_only_outputs=True)
    print(resposne)
    st.write("Reply:", resposne["output_text"])
    
    
def main():
    st.set_page_config("Chat PDF")
    st.header("Finance Chatbot💁")

    user_question = st.text_input("Ask your financial doubt here:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:  # Ensure files are uploaded
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)  # Pass the list of files
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()