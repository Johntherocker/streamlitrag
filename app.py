import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
import google.generativeai as genai
import tiktoken

import requests
from pathlib import Path

def download_file(url, local_path):
    if not local_path.exists():
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

def download_faiss_index():
    faiss_url = "https://www.dropbox.com/scl/fi/y5kw2ut5orxhww7a6pydp/index.faiss?rlkey=igd39gip391qvbutmil5hw3u8&st=eia4eke7&dl=1"
    pkl_url = "https://www.dropbox.com/scl/fi/kly0btqxm9tfd02wnxzds/index.pkl?rlkey=3tg1bdyemk4wx74htbjzct21a&st=ug8w4ebt&dl=1"

    index_faiss_path = Path("/tmp/index.faiss")
    index_pkl_path = Path("/tmp/index.pkl")

    if not index_faiss_path.exists():
        r = requests.get(faiss_url)
        r.raise_for_status()
        with open(index_faiss_path, "wb") as f:
            f.write(r.content)

    if not index_pkl_path.exists():
        r = requests.get(pkl_url)
        r.raise_for_status()
        with open(index_pkl_path, "wb") as f:
            f.write(r.content)

    return index_faiss_path.parent


openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] 
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Get current working directory in Jupyter
current_dir = os.getcwd()

faiss_folder_path = os.path.join(current_dir, "faiss_index_store")

# Load vector store and embeddings
@st.cache_resource
def load_faiss_index():
    embeddings = OpenAIEmbeddings()
    faiss_folder_path = download_faiss_index()
    faiss_index = FAISS.load_local(
        str(faiss_folder_path), embeddings, allow_dangerous_deserialization=True
    )
    return faiss_index

faiss_index = load_faiss_index()

# Initialize Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.5-pro")

# RAG query function
def rag_query(query: str, top_k: int = 4):
    similar_docs = faiss_index.similarity_search(query, k=top_k)
    context = "\n\n".join(doc.page_content for doc in similar_docs)

    prompt = f"""You are a business oracle with access to the best business books ever in bookall.txt. You give excellent business advice like a management consultant on steroids.
Context: Advice for businesses.
{context}

Question: {query}
Answer:"""

    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("📘 The Business Oracle (RAG + Gemini)")

query = st.text_area("Enter your business question:", key="query_input")

top_k = st.slider("Number of relevant documents (top_k)", 1, 20, 4)

if st.button("Get Advice"):
    if query.strip():
        with st.spinner("Thinking..."):
            answer = rag_query(query, top_k)
            st.markdown("### 💡 Advice:")
            st.write(answer)
    else:
        st.warning("Please enter a question.")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def rag_query_with_history(query: str, chat_history: list, top_k: int = 4):
    similar_docs = faiss_index.similarity_search(query, k=top_k)
    context = "\n\n".join(doc.page_content for doc in similar_docs)

    # Format chat history as dialogue
    history_text = ""
    for i, (q, a) in enumerate(chat_history):
        history_text += f"User: {q}\nBusiness Oracle: {a}\n"

    prompt = f"""You are a business oracle with access to the best business books ever in bookall.txt. You give excellent business advice like a management consultant on steroids.
Context: Advice for businesses.
{context}

Conversation history:
{history_text}

User: {query}
Answer:"""

    response = gemini_model.generate_content(prompt)
    return response.text

# In your Streamlit app UI code:

st.title("📘 The Business Oracle (RAG + Gemini)")

query = st.text_area("Enter your business question:")

top_k = st.slider("Number of relevant documents (top_k)", 1, 20, 4)

if st.button("Get Advice"):
    if query.strip():
        with st.spinner("Thinking..."):
            answer = rag_query_with_history(query, st.session_state.chat_history, top_k)
            # Append current Q&A to chat history
            st.session_state.chat_history.append((query, answer))

            st.markdown("### 💡 Advice:")
            st.write(answer)
    else:
        st.warning("Please enter a question.")

# Optionally show previous Q&A for user reference:
if st.session_state.chat_history:
    st.markdown("### Conversation History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
        st.markdown("---")
