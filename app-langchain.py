# app-langchain.py

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

from PyPDF2 import PdfReader
from pathlib import Path
import docx
from pptx import Presentation
from PIL import Image
import requests

# LangChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

st.set_page_config(page_title="Ultra-lite Data Analysis + RAG", layout="wide")

# ================= Session State =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "df_active" not in st.session_state:
    st.session_state.df_active = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Data Analysis"

# ================= Helpers =================
def safe_describe(df: pd.DataFrame):
    """Statistik ringkas aman"""
    try:
        return df.describe(include="all").transpose()
    except Exception:
        return pd.DataFrame()

def detect_data_types(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return categorical_cols, numeric_cols

def detect_date_columns(df: pd.DataFrame):
    return [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]

def safe_read_file(file):
    """Ekstrak teks dari berbagai format"""
    name = file.name
    text = ""
    try:
        if name.endswith(".txt"):
            text = file.read().decode("utf-8")
        elif name.endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif name.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet)
                text += f"[{sheet}]\n" + df.to_csv(index=False) + "\n"
        elif name.endswith(".csv"):
            df = pd.read_csv(file)
            text = df.to_csv(index=False)
        elif name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif name.endswith(".pptx"):
            prs = Presentation(file)
            text = "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])
        elif name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            api_key = st.secrets.get("OCR_SPACE_API_KEY", "")
            if api_key:
                file.seek(0)
                resp = requests.post(
                    "https://api.ocr.space/parse/image",
                    files={"file": file},
                    data={"apikey": api_key, "language": "eng"}
                )
                data = resp.json()
                if "ParsedResults" in data and data["ParsedResults"]:
                    text = data["ParsedResults"][0].get("ParsedText", "")
        else:
            text = file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"⚠️ Gagal baca {name}: {e}")
    return text

def build_vectorstore(files):
    docs = []
    for f in files:
        f.seek(0)
        text = safe_read_file(f)
        if text:
            docs.append(Document(page_content=text, metadata={"source": f.name}))
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(split_docs, embeddings)

# ================= Tabs =================
tab1, tab2 = st.tabs(["📊 Data Analysis", "📚 RAG Advanced"])

# ====== TAB 1: Data Analysis ======
with tab1:
    st.session_state.active_tab = "Data Analysis"
    st.subheader("Upload dataset untuk analisis")

    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"], key="data_uploader")
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state.dfs = {"CSV": df}
        else:
            xls = pd.ExcelFile(uploaded_file)
            st.session_state.dfs = {s: pd.read_excel(uploaded_file, sheet_name=s) for s in xls.sheet_names}

        st.success("✅ File berhasil diupload")

        sheet_names = list(st.session_state.dfs.keys())
        selected_sheets = st.multiselect("Pilih Sheet", sheet_names, default=sheet_names[:1])
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

            # st.dataframe(df.head())
            st.session_state.df_active = df
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
            

# ====== TAB 2: RAG Advanced ======
with tab2:
    st.session_state.active_tab = "RAG Advanced"
    st.subheader("Upload dokumen multi-format")

    rag_files = st.file_uploader(
        "Upload dokumen (PDF, TXT, DOCX, PPTX, CSV, XLSX, Gambar)",
        type=["pdf", "txt", "docx", "pptx", "csv", "xls", "xlsx", "png", "jpg", "jpeg", "bmp", "gif"],
        accept_multiple_files=True,
        key="rag_uploader"
    )
    if rag_files:
        st.session_state.vectorstore = build_vectorstore(rag_files)
        if st.session_state.vectorstore:
            st.success("✅ Vectorstore berhasil dibuat")
        else:
            st.error("❌ Gagal membangun vectorstore")

# ================= Chatbot (Global Root) =================
st.markdown("---")
st.subheader("💬 Chatbot")

# tampilkan riwayat chat
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)

user_query = st.chat_input("Tanyakan sesuatu...")
if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append(("user", user_query))
    response = "⚠️ Tidak ada jawaban."

    # --- Data Analysis Chatbot ---
    if st.session_state.active_tab == "Data Analysis":
        if st.session_state.df_active is None:
            response = "⚠️ Silakan upload dataset di tab Data Analysis dulu."
        else:
            df = st.session_state.df_active
            query_lower = user_query.lower()

            # Statistik
            if "statistik" in query_lower:
                stats = safe_describe(df)
                if not stats.empty:
                    st.dataframe(stats)
                    if "Sales" in df.columns:
                        sales_summary = df["Sales"].describe()
                        response = (
                            "📊 Statistik dataset sudah ditampilkan.\n\n"
                            f"- Jumlah baris: {df.shape[0]}\n"
                            f"- Jumlah kolom: {df.shape[1]}\n"
                            f"- Rata-rata Sales: {sales_summary['mean']:.2f}\n"
                            f"- Min: {sales_summary['min']:.2f}, Max: {sales_summary['max']:.2f}"
                        )
                    else:
                        response = f"📊 Statistik dataset ditampilkan ({df.shape[0]} baris, {df.shape[1]} kolom)."

            # Trend
            elif "trend" in query_lower or "grafik" in query_lower:
                date_cols = detect_date_columns(df)
                if date_cols and "Sales" in df.columns:
                    date_col = date_cols[0]
                    trend_df = df.groupby(date_col)["Sales"].sum().reset_index()
                    fig = px.line(trend_df, x=date_col, y="Sales", title="📈 Trend Sales")
                    st.plotly_chart(fig, use_container_width=True)
                    response = f"📈 Grafik trend Sales berdasarkan `{date_col}` sudah ditampilkan."
                else:
                    response = "⚠️ Tidak ada kolom tanggal atau Sales untuk membuat trend."

            # Kategori
            elif "kategori" in query_lower or "bar" in query_lower or "pie" in query_lower:
                categorical_cols, _ = detect_data_types(df)
                if categorical_cols and "Sales" in df.columns:
                    col = categorical_cols[0]
                    cat_df = df.groupby(col)["Sales"].sum().reset_index()
                    fig = px.bar(cat_df, x=col, y="Sales", title=f"📊 Total Sales per {col}")
                    st.plotly_chart(fig, use_container_width=True)
                    response = f"📊 Grafik kategori berdasarkan `{col}` sudah ditampilkan."
                else:
                    response = "⚠️ Tidak ada kolom kategori + Sales untuk membuat grafik kategori."

            # Default
            else:
                response = f"🔍 Dataset berisi {df.shape[0]} baris dan {df.shape[1]} kolom."

    # --- RAG Advanced Chatbot ---
    elif st.session_state.active_tab == "RAG Advanced":
        if not st.session_state.vectorstore:
            response = "⚠️ Silakan upload dokumen di tab RAG Advanced dulu."
        else:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            llm = HuggingFacePipeline.from_model_id(
                model_id="google/flan-t5-small",
                task="text2text-generation"
            )
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            result = qa({"query": user_query})
            answer = result["result"]
            sources = result.get("source_documents", [])
            response = answer
            if sources:
                response += "\n\n**Sumber:**\n"
                for s in sources:
                    response += f"- {s.metadata.get('source', '')}\n"

    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append(("assistant", response))
