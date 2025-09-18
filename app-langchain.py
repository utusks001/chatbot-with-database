# app-langchain.py

import streamlit as st
import os, io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests

# LangChain & RAG
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# File loaders
from PyPDF2 import PdfReader
import docx
from pptx import Presentation
from PIL import Image

# ====== Load dotenv kalau lokal ======
from dotenv import load_dotenv
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

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

def detect_data_types(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numeric_cols

# ====== OCR.Space Helper ======
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")

def ocr_image(file):
    text = ""
    file.seek(0)
    if OCR_SPACE_API_KEY:
        try:
            resp = requests.post(
                "https://api.ocr.space/parse/image",
                files={"file": file},
                data={"apikey": OCR_SPACE_API_KEY, "language": "eng"}
            )
            data = resp.json()
            if "ParsedResults" in data and data["ParsedResults"]:
                text = data["ParsedResults"][0].get("ParsedText", "")
        except Exception as e:
            st.warning(f"⚠️ OCR.Space error: {e}")
    return text

# ====== Build Vectorstore ======
def build_vectorstore(files):
    docs = []
    for file in files:
        name = file.name
        try:
            if name.endswith(".txt"):
                text = file.read().decode("utf-8")
            elif name.endswith(".pdf"):
                reader = PdfReader(file)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
            elif name.endswith(".csv"):
                df = pd.read_csv(file)
                text = df.to_csv(index=False)
            elif name.endswith((".xlsx", ".xls")):
                xls = pd.ExcelFile(file)
                text = ""
                for sheet in xls.sheet_names:
                    df = pd.read_excel(file, sheet_name=sheet)
                    text += f"[{sheet}]\n" + df.to_csv(index=False) + "\n"
            elif name.endswith(".docx"):
                doc = docx.Document(file)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif name.endswith(".pptx"):
                prs = Presentation(file)
                text = "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])
            elif name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                text = ocr_image(file)
            else:
                text = file.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": name}))
        except Exception as e:
            st.warning(f"⚠️ Gagal load file {name}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# ====== Main App ======
st.set_page_config(page_title="Data & Document RAG Chatbot", layout="wide")
st.title("📊🤖 Chatbot Analisis Data & Dokumen (RAG + HuggingFace)")

tab1, tab2 = st.tabs(["📊 Data Analysis", "📑 RAG Advanced"])

# ===== MODE 1: Data Analysis =====
with tab1:
    uploaded_file = st.file_uploader(
        "Upload file Excel/CSV untuk analisa data",
        type=["csv", "xls", "xlsx"]
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state.current_df = df
        else:
            xls = pd.ExcelFile(uploaded_file)
            st.session_state.current_df = pd.concat(
                [pd.read_excel(uploaded_file, sheet_name=sheet) for sheet in xls.sheet_names],
                ignore_index=True
            )

        st.dataframe(st.session_state.current_df.head(10))
        categorical_cols, numeric_cols = detect_data_types(st.session_state.current_df)
        st.write(f"Kolom Numerik: {numeric_cols}")
        st.write(f"Kolom Kategorikal: {categorical_cols}")
        st.text(df_info_text(st.session_state.current_df))
        st.write(f"**Data shape:** {st.session_state.current_df.shape}")
        st.dataframe(safe_describe(st.session_state.current_df))

        num_df = st.session_state.current_df.select_dtypes(include="number")
        if not num_df.empty:
            st.write("**Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

# ===== MODE 2: RAG Advanced =====
with tab2:
    uploaded_files = st.file_uploader(
        "Upload dokumen (PDF, TXT, DOCX, PPTX, CSV, XLSX, gambar) → bisa multi-file",
        type=["pdf", "txt", "docx", "pptx", "csv", "xls", "xlsx", "png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown("### 📂 Dokumen yang diupload:")
        for f in uploaded_files:
            st.write("- " + f.name)

        vectorstore = build_vectorstore(uploaded_files)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # LLM HuggingFace fallback
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-small",
            task="text2text-generation"
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        st.success("✅ Chatbot RAG siap digunakan")

# ====== Chatbot untuk Data Analysis & RAG =====
st.subheader("💬 Chat (Data Analysis + RAG)")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Tanyakan sesuatu tentang data atau dokumen...")

if user_query:
    st.chat_message("user").markdown(user_query)

    response_text = ""

    # Data Analysis fallback
    if "current_df" in st.session_state and st.session_state.current_df is not None:
        preview = st.session_state.current_df.head(1000).to_csv(index=False)
        prompt = f"Analisa data berikut dan buat insight utama + kesimpulan:\n{preview}\nPertanyaan: {user_query}"
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-small",
            task="text2text-generation"
        )
        response_text = llm(prompt)

    # RAG fallback
    elif uploaded_files:
        result = qa_chain({"query": user_query})
        response_text = result["result"]

    st.chat_message("assistant").markdown(response_text)
    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("assistant", response_text))

# Render chat history
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)
