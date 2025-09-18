# app-multisheet-rag.py

import streamlit as st
import os, io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import toml
from dotenv import load_dotenv, set_key

# LangChain & RAG
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# PDF / OCR
from PyPDF2 import PdfReader

# Load dotenv kalau lokal
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# ====== Helper deteksi local vs Streamlit Cloud ======
def is_streamlit_cloud():
    return os.environ.get("STREAMLIT_RUNTIME") is not None

# ====== API Key Management ======
with st.sidebar:
    st.header("🔑 Konfigurasi API Key")

    GOOGLE_API_KEY = (
        st.secrets.get("GOOGLE_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
    )

    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = st.text_input(
            "Masukkan GOOGLE_API_KEY (buat di https://aistudio.google.com/apikey)",
            type="password"
        )
        if GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            st.success("✅ GOOGLE_API_KEY berhasil dimasukkan")

            if is_streamlit_cloud():
                st.info("ℹ️ Running di Streamlit Cloud → gunakan menu Settings → Secrets untuk simpan permanen")
            else:
                # Save ke .env lokal
                set_key(dotenv_path, "GOOGLE_API_KEY", GOOGLE_API_KEY)
                st.success("✅ API Key disimpan ke .env (lokal)")
    else:
        st.success("✅ GOOGLE_API_KEY berhasil dimuat")

    # Embeddings toggle
    st.subheader("⚙️ Embeddings Settings")
    use_hf_embeddings = st.checkbox("Pakai HuggingFace embeddings saja", value=False)
    st.session_state["USE_HF_EMBEDDINGS"] = use_hf_embeddings

# ====== Helper Functions ======
def safe_describe(df: pd.DataFrame):
    try:
        return df.describe(include="all").transpose()
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def df_info_text(df: pd.DataFrame):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

# ====== Build Vectorstore ======
def build_vectorstore(file):
    """Bangun vectorstore dari dokumen upload (txt, pdf, csv, xlsx)."""

    docs = []

    if file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
        docs = [Document(page_content=text)]

    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        docs = [Document(page_content=text)]

    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        text = df.to_csv(index=False)
        docs = [Document(page_content=text)]

    elif file.name.endswith(".xlsx"):
        xls = pd.ExcelFile(file)
        text = "\n".join(
            [f"[{sheet}]\n" + pd.read_excel(file, sheet_name=sheet).to_csv(index=False) for sheet in xls.sheet_names]
        )
        docs = [Document(page_content=text)]

    else:
        docs = [Document(page_content=file.read().decode("utf-8", errors="ignore"))]

    # Split dokumen
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # ====== Embeddings Selection ======
    if st.session_state.get("USE_HF_EMBEDDINGS", False):
        st.info("ℹ️ Mode HuggingFace embeddings dipilih (manual).")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(split_docs, embeddings)

    # Default: coba Gemini embeddings → fallback HuggingFace
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY", "")
        )
        return FAISS.from_documents(split_docs, embeddings)
    except Exception as e:
        st.warning(f"⚠️ Gagal pakai Gemini embeddings ({e}). Fallback ke HuggingFace.")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(split_docs, embeddings)

# ====== Main App ======
st.set_page_config(page_title="Data & Document RAG Chatbot", layout="wide")
st.title("📑🤖 Chatbot Analisis Data & Dokumen (RAG + Gemini/HuggingFace)")

# Upload file
uploaded_file = st.file_uploader(
    "Upload file dokumen (Excel, CSV, PDF, TXT)", 
    type=["csv", "xls", "xlsx", "pdf", "txt"]
)

# ====== Jalankan jika ada file ======
if uploaded_file is not None:
    st.markdown(f"### 📄 Analisa: {uploaded_file.name}")

    # Bangun vectorstore
    vectorstore = build_vectorstore(uploaded_file)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    if "qa_chain" not in st.session_state:
        if os.getenv("GOOGLE_API_KEY"):
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        else:
            st.stop()

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        st.success("✅ Chatbot RAG siap digunakan")

    # Chat input
    user_query = st.chat_input("Tanyakan sesuatu tentang dokumen hukum/medis yang diupload...")
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("🔎 Menganalisis dokumen..."):
            try:
                result = st.session_state.qa_chain({"query": user_query})
                answer = result["result"]
                sources = result.get("source_documents", [])

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if sources:
                        st.write("**Sumber:**")
                        for s in sources:
                            st.caption(s.page_content[:200] + "...")
            except Exception as e:
                st.error(f"❌ Error saat menjawab: {e}")
