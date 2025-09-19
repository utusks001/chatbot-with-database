# app-langchain.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import os

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader, UnstructuredImageLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA


# ======================
# Utility Functions
# ======================
def df_info_text(df: pd.DataFrame) -> str:
    """Return summary info mirip df.info() tapi dalam bentuk string."""
    buffer = []
    buffer.append(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    buffer.append("Kolom dan tipe data:")
    for col in df.columns:
        buffer.append(f" - {col}: {df[col].dtype}, nulls: {df[col].isnull().sum()}")
    return "\n".join(buffer)


def safe_describe(df: pd.DataFrame):
    try:
        return df.describe(include="all")
    except Exception:
        return df.describe()


def detect_data_types(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numeric_cols


def plot_trend(df, time_col, value_col):
    fig = px.line(df, x=time_col, y=value_col, title=f"Trend {value_col} over {time_col}")
    return fig


def plot_category(df, cat_col, value_col):
    fig = px.bar(
        df.groupby(cat_col)[value_col].sum().reset_index(),
        x=cat_col, y=value_col, title=f"{value_col} by {cat_col}"
    )
    return fig


# ======================
# Streamlit App
# ======================
st.set_page_config(page_title="Chatbot Data Analysis + RAG", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Data Analysis"
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

st.title("🤖 Chatbot Data Analysis + RAG Advanced")

tab1, tab2 = st.tabs(["📊 Data Analysis", "📂 RAG Advanced"])

# ====== MODE 1: Data Analysis ======
with tab1:
    uploaded_file = st.file_uploader("Upload file Excel/CSV untuk analisa data", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    df = None
    if "uploaded_file" in st.session_state:
        f = st.session_state.uploaded_file
        if f.name.endswith(".csv"):
            df = pd.read_csv(f)
            st.session_state.dfs = {"CSV": df}
        else:
            xls = pd.ExcelFile(f)
            st.session_state.dfs = {sheet: pd.read_excel(f, sheet_name=sheet) for sheet in xls.sheet_names}

        st.subheader("📑 Pilih Sheet")
        sheet_names = list(st.session_state.dfs.keys())
        selected_sheets = st.multiselect("Sheet Aktif", sheet_names, default=sheet_names[:1])

        if selected_sheets:
            if len(selected_sheets) == 1:
                df = st.session_state.dfs[selected_sheets[0]]
            else:
                df_list = []
                for s in selected_sheets:
                    temp = st.session_state.dfs[s].copy()
                    temp["SheetName"] = s
                    df_list.append(temp)
                df = pd.concat(df_list, ignore_index=True)

            st.dataframe(df.head(10))
            categorical_cols, numeric_cols = detect_data_types(df)
            st.write(f"Kolom Numerik: {numeric_cols}")
            st.write(f"Kolom Kategorikal: {categorical_cols}")
            st.text(df_info_text(df))
            st.write(f"**Data shape:** {df.shape}")
            st.dataframe(safe_describe(df))

            if not df.select_dtypes(include="number").empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

    st.session_state.active_tab = "Data Analysis"

# ====== MODE 2: RAG Advanced ======
with tab2:
    rag_files = st.file_uploader(
        "Upload dokumen untuk RAG (PDF, TXT, DOCX, PPTX, Images)", 
        type=["pdf", "txt", "docx", "pptx", "jpg", "jpeg", "png", "gif", "bmp"],
        accept_multiple_files=True
    )
    if rag_files:
        docs = []
        for file in rag_files:
            ext = os.path.splitext(file.name)[1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(file)
            elif ext == ".txt":
                loader = TextLoader(file)
            elif ext == ".docx":
                loader = Docx2txtLoader(file)
            elif ext == ".pptx":
                loader = UnstructuredPowerPointLoader(file)
            elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                loader = UnstructuredImageLoader(file)
            else:
                continue
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0, "max_length": 512})
        st.session_state.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

    st.session_state.active_tab = "RAG Advanced"


# ======================
# Chatbot (Root-level)
# ======================
user_query = st.chat_input("Tanyakan sesuatu...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    response = ""
    extra_viz = None

    if st.session_state.active_tab == "Data Analysis":
        if "dfs" in st.session_state and st.session_state.dfs:
            df = list(st.session_state.dfs.values())[0]
            categorical_cols, numeric_cols = detect_data_types(df)

            if "trend" in user_query.lower() and any("date" in c.lower() for c in df.columns):
                time_col = [c for c in df.columns if "date" in c.lower()][0]
                value_col = numeric_cols[0] if numeric_cols else None
                if value_col:
                    response = f"Berikut tren {value_col} terhadap {time_col}."
                    extra_viz = plot_trend(df, time_col, value_col)

            elif "kategori" in user_query.lower() and categorical_cols:
                cat_col = categorical_cols[0]
                value_col = numeric_cols[0] if numeric_cols else None
                if value_col:
                    response = f"Berikut distribusi {value_col} berdasarkan {cat_col}."
                    extra_viz = plot_category(df, cat_col, value_col)

            else:
                response = f"Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom. Statistik dasar ditampilkan di atas."
        else:
            response = "⚠️ Silakan upload dataset di tab Data Analysis dulu."

    elif st.session_state.active_tab == "RAG Advanced":
        if st.session_state.rag_chain:
            res = st.session_state.rag_chain({"query": user_query})
            response = res["result"]
        else:
            response = "⚠️ Silakan upload dokumen di tab RAG Advanced dulu."

    else:
        response = "⚠️ Pilih tab Data Analysis atau RAG Advanced."

    st.session_state.chat_history.append({"role": "assistant", "content": response})

# tampilkan riwayat chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# tampilkan visualisasi tambahan (di luar chat)
if 'extra_viz' in locals() and extra_viz is not None:
    st.plotly_chart(extra_viz, use_container_width=True)
