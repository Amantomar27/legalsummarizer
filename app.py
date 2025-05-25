import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from ingest import ingest_documents
from prompts import CLAUSE_ANALYSIS_PROMPT
from dotenv import load_dotenv
import os

load_dotenv()
st.set_page_config(page_title="Legal Clause Checker", layout="wide")
st.title("⚖️ Legal Document Analyzer & Clause Checker")

uploaded_file = st.file_uploader("📄 Upload a contract PDF", type="pdf")

if uploaded_file:
    file_path = f"data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    with st.spinner("📚 Ingesting and analyzing document..."):
        vectordb = ingest_documents(file_path)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        llm = OpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    question = st.text_input("🔍 Ask about a clause, e.g., 'Explain termination clause'")

    if question:
        result = qa_chain.run(CLAUSE_ANALYSIS_PROMPT.format(text=question))
        st.markdown("### ✅ AI Legal Insight:")
        st.write(result)
