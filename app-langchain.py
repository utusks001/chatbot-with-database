# app-langchain.py

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile, os

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ====== INIT ======
st.set_page_config(page_title="📊 Data Analysis + 📚 RAG Assistant", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "rag_qa" not in st.session_state:
    st.session_state.rag_qa = None

# HuggingFace LLM (fallback model ringan)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature":0.1, "max_length":512}
)

# ====== UTILS ======
def safe_describe(df):
    try:
        return df.describe(include="all")
    except Exception:
        return df.describe()

def detect_data_types(df):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return categorical_cols, numeric_cols

def df_info_text(df):
    buf = []
    buf.append(f"Jumlah baris: {df.shape[0]}")
    buf.append(f"Jumlah kolom: {df.shape[1]}")
    buf.append("Tipe kolom:")
    buf.append(str(df.dtypes))
    return "\n".join(buf)

def generate_insight_with_llm(df, x_axis, y_axis):
    """Ambil subset data, kirim ke LLM untuk insight naratif"""
    try:
        subset = df[[x_axis, y_axis]].dropna().head(50)  # hanya sample biar ringan
        prompt = PromptTemplate.from_template(
            "Berikan insight ringkas dalam bahasa alami dari data berikut (format tabel):\n\n{data}\n\n"
            "Fokus pada tren, pola, anomali, dan kesimpulan utama."
        )
        insight = llm(prompt.format(data=subset.to_string()))
        return str(insight)
    except Exception as e:
        return f"⚠️ Gagal membuat insight otomatis: {e}"

# ====== LAYOUT ======
tab1, tab2 = st.tabs(["📊 Data Analysis", "📚 RAG Advanced"])

# ====== MODE 1: DATA ANALYSIS ======
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

            # Pilihan sumbu untuk tren
            st.subheader("📈 Buat Grafik Tren")
            x_axis = st.selectbox("Pilih kolom X (axis)", df.columns, index=0)
            y_axis = st.selectbox("Pilih kolom Y (axis)", df.columns, index=1)

            if st.button("Tampilkan Grafik"):
                if x_axis and y_axis:
                    trend_df = df[[x_axis, y_axis]].dropna().sort_values(x_axis)
                    if pd.api.types.is_datetime64_any_dtype(trend_df[x_axis]):
                        trend_df = trend_df.groupby(x_axis, as_index=False)[y_axis].sum()
                    fig = px.line(trend_df, x=x_axis, y=y_axis, title=f"Grafik tren {y_axis} vs {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)

                    # Insight otomatis dengan LLM
                    insight = generate_insight_with_llm(trend_df, x_axis, y_axis)
                    st.success(f"**Insight:** {insight}")

# ====== MODE 2: RAG ADVANCED ======
with tab2:
    rag_files = st.file_uploader(
        "Upload dokumen (PDF, TXT, DOCX, PPTX, Image)",
        type=["pdf","txt","docx","pptx","jpg","jpeg","png","bmp","gif"],
        accept_multiple_files=True
    )
    if rag_files:
        docs = []
        for file in rag_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(tmp_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(tmp_path)
            elif file.name.endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(tmp_path)
            else:
                loader = UnstructuredImageLoader(tmp_path)
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        st.session_state.rag_qa = RetrievalQA.from_chain_type(
            llm=llm, retriever=vectorstore.as_retriever()
        )
        st.success("✅ RAG siap digunakan!")

# ====== CHATBOT ROOT ======
st.subheader("💬 Chatbot")

user_query = st.chat_input("Tanyakan sesuatu...")
if user_query:
    st.session_state.chat_history.append(("user", user_query))

    response = "⚠️ Tidak ada konteks."
    # Jika tab Data Analysis aktif
    if st.session_state.get("uploaded_file") and df is not None and tab1:
        # Jawaban berbasis dataset
        if "statistik" in user_query.lower():
            response = str(safe_describe(df))
        elif "tren" in user_query.lower():
            response = "📈 Gunakan fitur grafik tren di atas untuk melihat pola visual."
        elif "kategori" in user_query.lower():
            cats, _ = detect_data_types(df)
            response = f"Kolom kategorikal yang tersedia: {cats}"
        else:
            # Pertanyaan umum dataset → kirim ringkasan ke LLM
            sample = df.head(50).to_string()
            prompt = PromptTemplate.from_template(
                "Jawab pertanyaan berikut berdasarkan dataset:\n\nPertanyaan: {q}\n\nData:\n{data}"
            )
            response = llm(prompt.format(q=user_query, data=sample))

    # Jika tab RAG Advanced aktif
    elif st.session_state.get("rag_qa"):
        response = st.session_state.rag_qa.run(user_query)

    st.session_state.chat_history.append(("assistant", response))

# Tampilkan chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
