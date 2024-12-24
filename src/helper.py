import PyPDF2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline
from src.prompt import *


PROMPT = PromptTemplate(template=prompt_template, input_veriables=['context','question'])
chain_type_kwargs={"prompt":PROMPT}
embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"

def extract_text_from_pdfs(pdf_folder):
    text = ""
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
    return text

def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def create_vector_database(chunks, embeddings_model):
    save_path = r"D:/AI-Projects/hydrogen-chatbot/vectore_db/vector_db.faiss"
    
    if os.path.exists(save_path):
        print(f"Loading existing vector database from {save_path}...")
        vector_db = FAISS.load_local(save_path, HuggingFaceEmbeddings(model_name=embeddings_model),allow_dangerous_deserialization=True)
    else:
        if not chunks:
            raise ValueError("Chunks are required to create a new vector database.")
        print("Creating a new vector database...")
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        vector_db = FAISS.from_texts(chunks, embeddings)
        vector_db.save_local(save_path)
        print(f"Vector database created and saved to {save_path}")
    
    return vector_db

def load_db():
    save_path = r"D:/AI-Projects/hydrogen-chatbot/vectore_db/vector_db.faiss"
    vector_db = FAISS.load_local(save_path, HuggingFaceEmbeddings(model_name=embeddings_model),allow_dangerous_deserialization=True)
    return vector_db





def load_pretrained_model():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

    # Define model_kwargs
    model_kwargs = {
     "max_length": 128
    }

    llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    token="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",
    temperature=0.7, 
    model_kwargs=model_kwargs)
    return llm 

def chat_bot(vector_db,llm,chain_type_kwargs):
    qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_db.as_retriever(search_kwargs={'k': 2,'score_threshold': 0.7},search_type="similarity_score_threshold"),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

    return qa