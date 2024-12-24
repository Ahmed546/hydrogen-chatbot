import PyPDF2
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline
from src.helper import load_db, load_pretrained_model, chat_bot, chain_type_kwargs
import streamlit as st

# Load database and model synchronously
db = load_db()
llm = load_pretrained_model()
qa = chat_bot(db, llm, chain_type_kwargs)

# Streamlit input for user question
user_input = st.text_input("Your question:")

# Button to get response
if st.button("Get Answer"):
    if user_input:
        result = qa.invoke({"query": user_input})
        st.write("Response:", result["result"])
    else:
        st.warning("Please enter a question.")

# Initialize chat history in session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Update chat history
if user_input:
    st.session_state.history.append(f"You: {user_input}")
if 'result' in locals():
    st.session_state.history.append(f"Chatbot: {result['result']}")

# Display chat history
if st.session_state.history:
    st.write("### Chat History")
    for message in st.session_state.history:
        st.write(message)

if __name__ == '__main__':
    # Streamlit runs the script from top to bottom on each interaction
    pass  # All code is executed during the initial run
