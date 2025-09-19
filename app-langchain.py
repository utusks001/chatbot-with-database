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

# ================= Config =================
st.set_page_config(page_title="Ultra-lite Data Analysis + RAG", layout="wide")

# ================= Session State Init =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Data Analysis"
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "active_df" not in st.session_state:
    st.session_state.active_df = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ================= Helper =================
def detect_data_types(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    return cat_cols, num_cols

def df_info_text(df):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

def safe_describe(df):
    return df.describe(include="all").T.fillna("")

def plot_trend(df, time_col, value_col):
    fig = px.line(df.sort_values(time_col), x=time_col, y=value_col, title=f"Trend {value_col} terhadap {time_col}")
    st.plotly_chart(fig, use_container_width=True)

def plot_category(df, cat_col, value_col):
    fig = px.bar(df, x=cat_col, y=value_col, title=f"Distribusi {value_col} berdasarkan {cat_col}")
    st.plotly_chart(fig, use_container_width=True)

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
            text = ""
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
            text = "\n".join(
                [shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")]
            )
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

# ====== MODE 1: Data Analysis ======
with tab1:
    st.session_state.active_tab = "Data Analysis"
    uploaded_file = st.file_uploader("Upload file Excel/CSV untuk analisa data", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

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

            # simpan df aktif
            st.session_state.active_df = df

            # tampilkan preview + info
            st.dataframe(df.head(10))
            categorical_cols, numeric_cols = detect_data_types(df)
            st.write(f"Kolom Numerik: {numeric_cols}")
            st.write(f"Kolom Kategorikal: {categorical_cols}")
            st.text(df_info_text(df))
            st.dataframe(safe_describe(df))

            if not df.select_dtypes(include="number").empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

# ====== MODE 2: RAG Advanced ======
with tab2:
    st.session_state.active_tab = "RAG Advanced"
    rag_files = st.file_uploader(
        "Upload dokumen (PDF, TXT, DOCX, PPTX, CSV, XLSX, Gambar)",
        type=["pdf", "txt", "docx", "pptx", "csv", "xls", "xlsx", "png", "jpg", "jpeg", "bmp", "gif"],
        accept_multiple_files=True,
    )
    if rag_files:
        st.session_state.vectorstore = build_vectorstore(rag_files)
        if st.session_state.vectorstore:
            st.success("✅ Vectorstore berhasil dibuat")
        else:
            st.error("❌ Gagal membangun vectorstore dari file yang diupload.")

# ================= Chatbot =================
st.markdown("---")
st.subheader("💬 Chatbot")

for h in st.session_state.chat_history:
    with st.chat_message(h["role"]):
        st.markdown(h["content"])

user_query = st.chat_input("Tanyakan sesuatu...")
if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    response = ""
    extra_viz = None

    # --- Data Analysis ---
    if st.session_state.active_tab == "Data Analysis":
        if st.session_state.active_df is not None:
            df = st.session_state.active_df
            categorical_cols, numeric_cols = detect_data_types(df)

            if "statistik" in user_query.lower() or "describe" in user_query.lower():
                response = "📊 Statistik dasar dataset ditampilkan di atas."

            elif "heatmap" in user_query.lower() or "korelasi" in user_query.lower():
                response = "🔥 Heatmap korelasi antar kolom numerik sudah ditampilkan."

            elif "trend" in user_query.lower():
                date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
                if date_cols and numeric_cols:
                    plot_trend(df, date_cols[0], numeric_cols[0])
                    response = f"📈 Grafik tren {numeric_cols[0]} terhadap {date_cols[0]} ditampilkan."
                elif numeric_cols:
                    plot_trend(df.reset_index(), "index", numeric_cols[0])
                    response = f"📈 Grafik tren {numeric_cols[0]} ditampilkan."
                else:
                    response = "⚠️ Tidak ada kolom numerik untuk membuat grafik trend."

            elif "kategori" in user_query.lower() and categorical_cols and numeric_cols:
                plot_category(df, categorical_cols[0], numeric_cols[0])
                response = f"📊 Distribusi {numeric_cols[0]} berdasarkan {categorical_cols[0]} ditampilkan."

            else:
                response = f"🔍 Dataset berisi {df.shape[0]} baris dan {df.shape[1]} kolom."

        else:
            response = "⚠️ Silakan upload dataset di tab Data Analysis dulu."

    # --- RAG Advanced ---
    elif st.session_state.active_tab == "RAG Advanced":
        if not st.session_state.vectorstore:
            response = "⚠️ Silakan upload dokumen di tab RAG Advanced dulu."
        else:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            llm = HuggingFacePipeline.from_model_id(
                model_id="google/flan-t5-small", task="text2text-generation"
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

    # tampilkan jawaban
    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
