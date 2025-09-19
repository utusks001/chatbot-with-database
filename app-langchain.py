# app-langchain.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, UnstructuredImageLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# ==============================
# Helper functions
# ==============================

def df_info_text(df: pd.DataFrame) -> str:
    """Generate basic dataset info text."""
    buf = []
    buf.append(f"Shape: {df.shape}")
    buf.append(f"Columns: {list(df.columns)}")
    buf.append("Types:")
    buf.append(str(df.dtypes))
    return "\n".join(buf)

def safe_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Describe numeric safely"""
    try:
        return df.describe(include="all").transpose()
    except Exception:
        return df.describe().transpose()

def detect_data_types(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return cat_cols, num_cols

def analyze_dataframe(df: pd.DataFrame, query: str):
    """Handle analysis queries: statistics, trend, category plots"""
    query_lower = query.lower()
    result = ""

    if any(kw in query_lower for kw in ["statistik", "summary", "deskripsi", "describe"]):
        result += "📊 Statistik dasar dataset:\n"
        st.dataframe(safe_describe(df))

    if any(kw in query_lower for kw in ["trend", "tren", "waktu", "time", "sales", "penjualan"]):
        date_cols = [c for c in df.columns if "date" in c.lower()]
        val_cols = [c for c in df.columns if c.lower() in ["sales", "profit", "amount"]]
        if date_cols and val_cols:
            x_col = date_cols[0]
            y_col = val_cols[0]
            fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, title=f"Trend {y_col} berdasarkan {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            result += f"📈 Grafik tren {y_col} terhadap {x_col} sudah dibuat.\n"

    if any(kw in query_lower for kw in ["kategori", "category", "bar", "pie"]):
        cat_cols, num_cols = detect_data_types(df)
        if cat_cols and num_cols:
            cat = cat_cols[0]
            num = num_cols[0]
            fig = px.bar(df.groupby(cat)[num].sum().reset_index(),
                         x=cat, y=num, title=f"Distribusi {num} berdasarkan {cat}")
            st.plotly_chart(fig, use_container_width=True)
            result += f"📊 Grafik kategori {cat} vs {num} sudah dibuat.\n"

    if not result:
        result = "ℹ️ Pertanyaan tidak cocok dengan data analysis, coba gunakan kata kunci statistik / tren / kategori."
    return result

def analyze_rag(vectorstore, query: str):
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0}),
        retriever=retriever
    )
    return qa.run(query)

def detect_mode(query: str, df, vectorstore):
    """Auto detect mode based on query + availability"""
    q = query.lower()
    dataset_keywords = ["statistik", "summary", "describe", "tren", "trend",
                        "sales", "penjualan", "kategori", "grafik", "kolom", "insight", "kesimpulan"]
    doc_keywords = ["dokumen", "isi file", "pdf", "halaman", "gambar", "image", "teks"]

    if any(k in q for k in dataset_keywords) and df is not None:
        return "data"
    elif any(k in q for k in doc_keywords) and vectorstore is not None:
        return "rag"
    elif df is not None and vectorstore is not None:
        return "ask"  # ambiguous → tanya balik user
    elif df is not None:
        return "data"
    elif vectorstore is not None:
        return "rag"
    else:
        return "none"

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="Data Analysis + RAG", layout="wide")

st.title("📊 Data Analysis + 📑 RAG Advanced")

tab1, tab2 = st.tabs(["Data Analysis", "RAG Advanced"])

# --- Tab 1: Data Analysis ---
with tab1:
    uploaded_file = st.file_uploader("Upload Excel/CSV untuk analisa", type=["csv", "xls", "xlsx"])
    df = None
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            xls = pd.ExcelFile(uploaded_file)
            df = pd.read_excel(uploaded_file, sheet_name=xls.sheet_names[0])
        st.session_state.df = df

    if "df" in st.session_state:
        df = st.session_state.df
        st.subheader("Preview Data")
        st.dataframe(df.head(10))
        cat_cols, num_cols = detect_data_types(df)
        st.write(f"Kolom Numerik: {num_cols}")
        st.write(f"Kolom Kategorikal: {cat_cols}")
        st.text(df_info_text(df))

# --- Tab 2: RAG Advanced ---
with tab2:
    uploaded_docs = st.file_uploader("Upload dokumen (PDF, TXT, DOCX, PPTX, Image)", 
                                     type=["pdf", "txt", "docx", "pptx", "png", "jpg", "jpeg", "bmp", "gif"],
                                     accept_multiple_files=True)
    if uploaded_docs:
        documents = []
        for file in uploaded_docs:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file)
            elif file.name.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file)
            elif file.name.endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(file)
            else:  # images
                loader = UnstructuredImageLoader(file)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.session_state.vectorstore = vectorstore
        st.success("✅ Dokumen berhasil diproses ke vectorstore")

# ==============================
# Chatbot Root (Auto Mode)
# ==============================
st.subheader("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# tampilkan riwayat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Tanyakan sesuatu...")
if user_query:
    st.session_state.messages.append({"role":"user","content":user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    df = st.session_state.get("df")
    vectorstore = st.session_state.get("vectorstore")
    mode = detect_mode(user_query, df, vectorstore)

    if mode == "data":
        answer = analyze_dataframe(df, user_query)
    elif mode == "rag":
        answer = analyze_rag(vectorstore, user_query)
    elif mode == "ask":
        answer = "🤔 Pertanyaan bisa dijawab dari Dataset maupun Dokumen. Mau pakai yang mana?"
    else:
        answer = "⚠️ Belum ada dataset atau dokumen yang diunggah."

    st.session_state.messages.append({"role":"assistant","content":answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
