import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document

# Load FAISS index and documents
with open("rag_index.pkl", "rb") as f:
    index, documents = pickle.load(f)

# Wrap documents
documents_dict = {str(i): Document(page_content=documents[i]) for i in range(len(documents))}
docstore = InMemoryDocstore(documents_dict)
index_to_docstore_id = {i: str(i) for i in range(len(documents))}

# Embedding model
embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Rebuild vectorstore
vectorstore = FAISS(
    embedding_function=embedding_fn,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Generation model
gen_pipeline = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",
    tokenizer="tiiuae/falcon-rw-1b",
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    pad_token_id=2  # avoids generation warning
)
llm = HuggingFacePipeline(pipeline=gen_pipeline)

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Streamlit UI
st.set_page_config(page_title="Loan Q&A Chatbot", layout="centered")
st.title("📊 Loan Approval Q&A Chatbot")
st.write("Ask any question about the loan dataset:")

query = st.text_input("Your question:")
if query:
    response = qa_chain.invoke({"query": query})
    st.write("💬", response["result"])
