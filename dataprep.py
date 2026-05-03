### This script processes the "KOBALIYA AND OTHERS v. RUSSIA.pdf" document, 
### splits it into chunks, generates embeddings using LegalBERT, 
### and saves a FAISS index for efficient retrieval. 
### It also saves the document chunks for use with BM25 retrieval in the app.

import os
import pickle

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModel
import torch


PDF_PATH = "KOBALIYA AND OTHERS v. RUSSIA.pdf"
INDEX_DIR = "indexes/kobaliya_faiss"
CHUNKS_PATH = "indexes/kobaliya_doc_chunks.pkl"

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
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean Pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings

    def __call__(self, text):
        return self.embed_query(text)


def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# Build and save index
def main():
    os.makedirs("indexes", exist_ok=True)

    print("Loading PDF...")
    documents = load_documents(PDF_PATH)

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    doc_chunks = text_splitter.split_documents(documents)

    print(f"Created {len(doc_chunks)} chunks.")

    print("Loading LegalBERT embeddings model...")
    embeddings = LegalBertEmbeddings()

    print("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)

    print("Saving FAISS index...")
    vectorstore.save_local(INDEX_DIR)

    print("Saving document chunks for BM25...")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(doc_chunks, f)

    print("Data preparation complete.")


if __name__ == "__main__":
    main()