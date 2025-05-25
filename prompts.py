CLAUSE_ANALYSIS_PROMPT = """
You are a legal assistant. Analyze the following clause from a contract.

1. Summarize what the clause means in plain English.
2. Identify if this clause is risky or contains problematic terms.
3. Suggest what might be missing (e.g., indemnity, jurisdiction, arbitration, NDA, etc.)
Clause:
\"\"\"{text}\"\"\"
"""
