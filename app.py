from urllib import response
import streamlit as st
import os
from getpass import getpass
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

LLM = ChatOpenAI(model="gpt-4o-mini")
VECTOR_STORE: Chroma | None = None 
PROMPT = PromptTemplate.from_template("""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
""")

def load_pdf_files(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return Document(page_content = text)

def documents_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.split_text(documents)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    global VECTOR_STORE
    VECTOR_STORE = Chroma.from_documents(text_chunks, embeddings=embeddings)
    

# def format_docs(docs)->str:
#     return "\n\n".join(doc.page_content for doc in docs)
    
def get_rag_chain():

    if VECTOR_STORE is None:
        raise os.error(500, "Vector store not found!")
    
    rag_chain = (
        {
            "context": VECTOR_STORE.as_retriever() | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | LLM
        | StrOutputParser()
    )
    return rag_chain


def main():
    st.set_page_config("Chat PDF") # title of the page
    st.header("Chat with PDF using OpenAI💁") # header of the page

    with st.sidebar: 
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = load_pdf_files(pdf_docs)
                text_chunks = documents_into_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    user_question = st.text_input("Ask a Question from the PDF Files") #widget

    if user_question:
        chain = get_rag_chain()
        response = chain.invoke(user_question)

        st.write(response)

if __name__ == "__main__":
    main()

