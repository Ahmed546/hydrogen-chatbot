prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, clearly state that you don't know, and avoid making up an answer.

Context: {context}
Question: {question}

Please provide a detailed and informative response based on the context given. If relevant, elaborate on key points to enhance understanding.

Only return the helpful answer below and nothing else.
Helpful answer:
"""