# General
import csv
from datetime import datetime
import numpy as np
import os
import tempfile

# Application  
import streamlit as st

# Textual Chains
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers import EnsembleRetriever, BM25Retriever

# NLP & Embeddings
from transformers import AutoTokenizer, AutoModel
import torch

import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load API key from environment variable
secret = os.getenv('OPENAI_API_KEY')

LOG_FILE = "chat_log.csv"

# --- Logging ---
def log_interaction(user_input: str, assistant_output: str):
    """Append timestamped chat interactions to a CSV log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Write header only if file is new
        if not file_exists:
            writer.writerow(["timestamp", "user_input", "assistant_output"])
        writer.writerow([timestamp, user_input, assistant_output])

# --- Retrieval related ---
class LegalBertEmbeddings:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean Pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings

    def __call__(self, text):
        return self.embed_query(text)

@st.cache_resource
def get_embeddings():
    return LegalBertEmbeddings()

def setup_vectorstore(documents):
    st.write("Setting up vectorstore...")
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstores = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstores, doc_chunks

def load_documents(file_path):
    loader = PyPDFLoader(
        file_path
    )
    documents = loader.load()
    return documents


# --- GenAI ---

def create_chain(vectorstores, doc_chunks):
    faiss_retriever = vectorstores.as_retriever(search_kwargs={'k':8})
    # Specify ranking function BM25 for added retriever
    bm25 = BM25Retriever.from_documents(doc_chunks)
    bm25.k = 5
    # Hybrid retriever combining FAISS and BM25
    retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25], weights=[0.5, 0.5])
    
    # Create a container for streaming
    container = st.empty()

    llm = ChatOpenAI(
        openai_api_key=secret,
        model="gpt-4.1-nano",
        temperature=0
    )

    # Reformulate follow-up questions
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer with retrieved docs
    qa_system_prompt = (
        "You are a helpful assistant that answers questions strictly based on the provided document content. "
        "Do not make up facts or add any background information that is not explicitly in the document. "
        "Conclude every answer by naturally integrating a critical analysis of the Russian foreign agent legislation, referencing one key International Human Rights document as part of your reasoning. Avoid explicitly labeling this section as 'Critical analysis'. \n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Combine into RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def main():
    st.set_page_config(
        page_title="Interact with EHRC Document",
        page_icon="📚",
        layout="centered"
    )
    
    st.image("images/EHRC_v2.jpg") 
    st.title("👩‍⚖️ Explore KOBALIYA AND OTHERS v. RUSSIA case 👨‍🎓")
    st.text("Application designed for interactive court summary Q&A.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    file_path = "KOBALIYA AND OTHERS v. RUSSIA.pdf"
    if "vectorstores" not in st.session_state or "doc_chunks" not in st.session_state:
        st.session_state.vectorstores, st.session_state.doc_chunks = setup_vectorstore(load_documents(file_path))
        st.success("Vector space created successfully!")

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(
            st.session_state.vectorstores, 
            st.session_state.doc_chunks
        )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask any questions related to case document")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = st.session_state.conversation_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            assistant_response = response["answer"]
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        
            # Log interaction
            try:
                log_interaction(user_input, assistant_response)
            except Exception as e:
                st.warning(f"⚠️ Logging failed: {e}")

if __name__ == '__main__':
    main()
