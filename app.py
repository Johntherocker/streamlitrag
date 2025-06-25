import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
import google.generativeai as genai

import requests
from pathlib import Path

def download_file(url, local_path):
    if not local_path.exists():
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

def download_faiss_index():
    base_url = "https://github.com/Johntherocker/streamlitrag/tree/master"
    download_file(base_url + "/index.faiss", Path("/tmp/index.faiss"))
    download_file(base_url + "/index.pkl", Path("/tmp/index.pkl"))


openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key 
os.environ["GOOGLE_API_KEY"] = google_api_key

# Get current working directory in Jupyter
current_dir = os.getcwd()

faiss_folder_path = os.path.join(current_dir, "faiss_index_store")

# Load vector store and embeddings
@st.cache_resource
def load_faiss_index():
    index_path = download_faiss_index()
    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS.load_local(str(index_path.parent), embeddings,allow_dangerous_deserialization=True)
    return faiss_index

faiss_index = load_faiss_index()

# Initialize Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.5-pro")

# RAG query function
def rag_query(query: str, top_k: int = 4):
    similar_docs = faiss_index.similarity_search(query, k=top_k)
    context = "\n\n".join(doc.page_content for doc in similar_docs)

    prompt = f"""You are a small business oracle with access to the best business books ever in bookall.txt. You give excellent business advice like a management consultant on steroids.
Context: Advice for small businesses
{context}

Question: {query}
Answer:"""

    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("📘 Small Business Advisor (RAG + Gemini)")

query = st.text_area("Enter your business question:")

top_k = st.slider("Number of relevant documents (top_k)", 1, 100, 4)

if st.button("Get Advice"):
    if query.strip():
        with st.spinner("Thinking..."):
            answer = rag_query(query, top_k)
            st.markdown("### 💡 Advice:")
            st.write(answer)
    else:
        st.warning("Please enter a question.")
