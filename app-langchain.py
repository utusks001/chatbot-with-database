# app-langchain.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader,
    UnstructuredImageLoader
)
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ===============================
# Helper Functions
# ===============================
def detect_data_types(df):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numeric_cols

def df_info_text(df):
    return f"🔹 Baris: {df.shape[0]}, Kolom: {df.shape[1]}\nKolom: {list(df.columns)}"

def safe_describe(df):
    try:
        return df.describe(include="all").transpose()
    except Exception:
        return df.describe().transpose()

def build_vectorstore(files):
    docs = []
    for file in files:
        name = file.name.lower()
        path = f"temp_{file.name}"
        with open(path, "wb") as f:
            f.write(file.read())

        if name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif name.endswith(".txt"):
            loader = TextLoader(path)
        elif name.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(path)
        elif name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            loader = UnstructuredImageLoader(path)
        else:
            continue
        docs.extend(loader.load())
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(texts, embeddings)
    return store

# ===============================
# Streamlit Layout
# ===============================
st.set_page_config(page_title="📊 Data Analysis + 📚 RAG Chatbot", layout="wide")

# State management
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "df" not in st.session_state:
    st.session_state.df = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_analysis" not in st.session_state:
    st.session_state.chat_analysis = []
if "chat_rag" not in st.session_state:
    st.session_state.chat_rag = []

st.title("🤖 Chatbot Data Analysis + RAG Advanced")

tab1, tab2 = st.tabs(["📊 Data Analysis", "📚 RAG Advanced"])

# ====== TAB 1: Data Analysis ======
with tab1:
    uploaded_file = st.file_uploader("Upload file Excel/CSV", type=["csv", "xls", "xlsx"], key="data_file")
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state.dfs = {"CSV": df}
        else:
            xls = pd.ExcelFile(uploaded_file)
            st.session_state.dfs = {s: pd.read_excel(uploaded_file, sheet_name=s) for s in xls.sheet_names}

        st.subheader("📑 Pilih Sheet")
        sheet_names = list(st.session_state.dfs.keys())
        selected_sheets = st.multiselect("Sheet Aktif", sheet_names, default=[sheet_names[0]])

        if selected_sheets:
            if len(selected_sheets) == 1:
                df = st.session_state.dfs[selected_sheets[0]]
            else:
                df = pd.concat([st.session_state.dfs[s] for s in selected_sheets], ignore_index=True)
            st.session_state.df = df

            st.dataframe(df.head(10))
            cat_cols, num_cols = detect_data_types(df)
            st.write(f"Kolom Numerik: {num_cols}")
            st.write(f"Kolom Kategorikal: {cat_cols}")
            st.text(df_info_text(df))
            st.write(f"**Data shape:** {df.shape}")
            st.dataframe(safe_describe(df))

            if not df.select_dtypes(include="number").empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

    # Chatbot untuk Data Analysis
    st.subheader("💬 Chatbot Data Analysis")
    user_query = st.chat_input("Tanyakan sesuatu tentang data...", key="chat_analysis_input")
    if user_query:
        st.session_state.chat_analysis.append(("user", user_query))
        df = st.session_state.df
        response = ""

        if df is not None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if "statistik" in user_query.lower():
                response = str(safe_describe(df))
            elif "trend" in user_query.lower() and "sales" in user_query.lower():
                if "Sales" in df.columns:
                    fig = px.line(df, y="Sales", title="Trend Penjualan")
                    st.plotly_chart(fig)
                    response = "📈 Grafik tren penjualan ditampilkan."
                else:
                    response = "⚠️ Kolom 'Sales' tidak ditemukan."
            elif "kategori" in user_query.lower():
                if cat_cols:
                    col_choice = st.selectbox("Pilih kolom kategori:", cat_cols, key="cat_choice")
                    fig = px.bar(df[col_choice].value_counts().reset_index(),
                                 x="index", y=col_choice, title=f"Distribusi {col_choice}")
                    st.plotly_chart(fig)
                    response = f"📊 Grafik kategori untuk {col_choice} ditampilkan."
                else:
                    response = "⚠️ Tidak ada kolom kategorikal."
            else:
                response = "✅ Data siap dianalisis. Silakan minta statistik, tren, atau grafik kategori."
        else:
            response = "⚠️ Silakan upload file di tab Data Analysis dulu."

        st.session_state.chat_analysis.append(("bot", response))

    for role, msg in st.session_state.chat_analysis:
        with st.chat_message(role):
            st.markdown(msg)

# ====== TAB 2: RAG Advanced ======
with tab2:
    rag_files = st.file_uploader(
        "Upload dokumen (PDF, TXT, DOCX, PPTX, Gambar)",
        type=["pdf", "txt", "docx", "pptx", "png", "jpg", "jpeg", "bmp", "gif"],
        accept_multiple_files=True,
        key="rag_files"
    )
    if rag_files:
        st.session_state.vectorstore = build_vectorstore(rag_files)
        st.success("✅ Vectorstore berhasil dibuat!")

    # Chatbot untuk RAG
    st.subheader("💬 Chatbot RAG Advanced")
    user_query = st.chat_input("Tanyakan sesuatu tentang dokumen...", key="chat_rag_input")
    if user_query:
        st.session_state.chat_rag.append(("user", user_query))
        response = ""

        if st.session_state.vectorstore:
            docs = st.session_state.vectorstore.similarity_search(user_query, k=2)
            response = "\n\n".join([d.page_content for d in docs])
        else:
            response = "⚠️ Silakan upload dokumen di tab RAG Advanced dulu."

        st.session_state.chat_rag.append(("bot", response))

    for role, msg in st.session_state.chat_rag:
        with st.chat_message(role):
            st.markdown(msg)
