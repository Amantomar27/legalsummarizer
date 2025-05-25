import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

from prompts import CLAUSE_ANALYSIS_PROMPT

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API Key not found. Please check your .env file.")
    st.stop()

# Initialize embedding and LLM
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key)

st.title("📄 Legal Document Analyzer & Clause Checker")
st.write("Upload a legal PDF, and we'll analyze its clauses and summarize risky content.")

uploaded_file = st.file_uploader("Upload a legal PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded file locally
    file_path = f"uploaded_files/{uploaded_file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded successfully!")

    # Load and split PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    # Vector store
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="db")
    vectordb.persist()

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": CLAUSE_ANALYSIS_PROMPT}
    )

    # Run analysis
    query = "Analyze this document for missing or risky clauses like indemnity, auto-renewal, arbitration, etc."
    with st.spinner("Analyzing document..."):
        result = qa_chain.run(query)

    # Show result
    st.subheader("📌 Clause Risk Summary")
    st.write(result)
