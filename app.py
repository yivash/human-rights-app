#Changes: 
# Read from created vectorstore and doc_chunks in session state, don't need to provide pdf

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
from datetime import datetime
import pickle

from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from transformers import AutoTokenizer, AutoModel
import torch

from datetime import datetime
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

load_dotenv()
#secret = os.getenv("OPENAI_API_KEY")

INDEX_DIR = "indexes/kobaliya_faiss"
CHUNKS_PATH = "indexes/kobaliya_doc_chunks.pkl"


def get_gsheet_worksheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    service_account_info = {
        "type": os.environ["GCP_TYPE"],
        "project_id": os.environ["GCP_PROJECT_ID"],
        "private_key_id": os.environ["GCP_PRIVATE_KEY_ID"],
        "private_key": os.environ["GCP_PRIVATE_KEY"], 
        "client_email": os.environ["GCP_CLIENT_EMAIL"],
        "client_id": os.environ["GCP_CLIENT_ID"],
        "auth_uri": os.environ["GCP_AUTH_URI"],
        "token_uri": os.environ["GCP_TOKEN_URI"],
        "auth_provider_x509_cert_url": os.environ["GCP_AUTH_PROVIDER_X509_CERT_URL"],
        "client_x509_cert_url": os.environ["GCP_CLIENT_X509_CERT_URL"],
    }

    credentials = Credentials.from_service_account_info(
        service_account_info,
        scopes=scopes,
    )

    client = gspread.authorize(credentials)

    spreadsheet = client.open(os.getenv("GSHEET_NAME"))
    worksheet = spreadsheet.worksheet(os.getenv("GSHEET_WORKSHEET"))

    return worksheet

def log_interaction(user_input: str, assistant_output: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    worksheet = get_gsheet_worksheet()

    worksheet.append_row(
        [
            timestamp,
            user_input,
            assistant_output,
        ],
        value_input_option="USER_ENTERED",
    )


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
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings

    def __call__(self, text):
        return self.embed_query(text)


@st.cache_resource
def get_embeddings():
    return LegalBertEmbeddings()


@st.cache_resource
def load_vectorstore_and_chunks():
    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    with open(CHUNKS_PATH, "rb") as f:
        doc_chunks = pickle.load(f)

    return vectorstore, doc_chunks

def create_chain(vectorstore, doc_chunks):
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    bm25 = BM25Retriever.from_documents(doc_chunks)
    bm25.k = 5

    retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25],
        weights=[0.5, 0.5]
    )

    llm = ChatOllama(
        model="gemma2:2b",
        temperature=0
    )

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
        llm,
        retriever,
        contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are a helpful assistant that answers questions strictly based on the provided document content. "
        "Do not make up facts or add any background information that is not explicitly in the document. "
        "Conclude every answer by naturally integrating a critical analysis of the Russian foreign agent legislation, "
        "referencing one key International Human Rights document as part of your reasoning. "
        "Avoid explicitly labeling this section as 'Critical analysis'.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    return rag_chain

def main():
    st.set_page_config(
        page_title="Human Rights AI Coach",
        page_icon="📚",
        layout="centered"
    )

    st.image("images/ehrc_logo_2.png", width=500)
    st.title("👩‍🏫 AI Coach")
    st.text("⚡Question foreign agents narrative and legislation in Russia \nwith KOBALIYA AND OTHERS v. RUSSIA court decision.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    vectorstore, doc_chunks = load_vectorstore_and_chunks()

    conversation_chain = create_chain(vectorstore, doc_chunks)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask any questions related to case document")

    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = conversation_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })

            assistant_response = response["answer"]
            st.markdown(assistant_response)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": assistant_response
            })

            try:
                log_interaction(user_input, assistant_response)
            except Exception as e:
                st.warning(f"⚠️ Logging failed: {e}")


if __name__ == "__main__":
    main()