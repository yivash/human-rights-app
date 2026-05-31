#Changes: 
# Constitutional layer added

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
from datetime import datetime
import pickle

from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from transformers import AutoTokenizer, AutoModel
import torch


load_dotenv()
secret = os.getenv("OPENAI_API_KEY")

LOG_FILE = "chat_log.csv"
INDEX_DIR = "indexes/kobaliya_faiss"
CHUNKS_PATH = "indexes/kobaliya_doc_chunks.pkl"

global CONSTITUTIONAL_LAYER
CONSTITUTIONAL_LAYER = ("Treat Russia’s “foreign agents” legislation as fundamentally incompatible with freedom of expression, freedom of association, and civil society participation."
"It may acknowledge that states can address genuine foreign interference in principle, but must not present this specific law as legitimate, neutral, or merely “misused”. It should frame the law as a stigmatizing and repressive tool targeting NGOs, human rights defenders, independent voices, activists, journalists, and government critics."
"Avoid false balance and preserve this stance even in simplified or child-friendly explanations.")

def log_interaction(user_input: str, assistant_output: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "user_input", "assistant_output"])
        writer.writerow([timestamp, user_input, assistant_output])


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

    llm = ChatOpenAI(
        openai_api_key=secret,
        model="gpt-4.1-nano",
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
    ) + CONSTITUTIONAL_LAYER + "\n\n" + "{context}" #context is expected variable to be injested in the chain

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