from langchain.prompts import PromptTemplate

CLAUSE_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
You are a legal assistant. Given the following context from a contract or legal document, perform the following tasks:
1. Summarize the key points.
2. Highlight risky clauses (e.g., auto-renewal, indemnity, unilateral termination).
3. Suggest any missing clauses that are commonly included in standard agreements.

Context:
{context}

Respond in bullet points.
"""
)
