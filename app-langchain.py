# app-langchain.py

import streamlit as st
import os, io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
from PIL import Image

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

# ===== Load dotenv kalau lokal =====
from dotenv import load_dotenv
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# ===== Helper Functions =====
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
    categorical_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numeric_cols

# ===== OCR.Space Helper =====
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")

def ocr_image(file):
    text = ""
    file.seek(0)
    if OCR_SPACE_API_KEY:
        try:
            resp = requests.post(
                "https://api.ocr.space/parse/image",
                files={"file": file},
                data={"apikey": OCR_SPACE_API_KEY, "language":"eng"}
            )
            data = resp.json()
            if "ParsedResults" in data and data["ParsedResults"]:
                text = data["ParsedResults"][0].get("ParsedText","")
        except Exception as e:
            st.warning(f"⚠️ OCR.Space error: {e}")
    return text

# ===== Build Vectorstore =====
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
            elif name.endswith((".xlsx",".xls")):
                xls = pd.ExcelFile(file)
                text = ""
                for sheet in xls.sheet_names:
                    df = pd.read_excel(file, sheet_name=sheet)
                    text += f"[{sheet}]\n"+df.to_csv(index=False)+"\n"
            elif name.endswith(".docx"):
                doc = docx.Document(file)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif name.endswith(".pptx"):
                prs = Presentation(file)
                text = "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape,"text")])
            elif name.lower().endswith((".png",".jpg",".jpeg",".bmp")):
                text = ocr_image(file)
            else:
                text = file.read().decode("utf-8",errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": name}))
        except Exception as e:
            st.warning(f"⚠️ Gagal load file {name}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# ===== Main App =====
st.set_page_config(page_title="Data & Document RAG Chatbot", layout="wide")
st.title("📊🤖 Chatbot Analisis Data & Dokumen (RAG + HF)")

tab1, tab2 = st.tabs(["📊 Data Analysis","📑 RAG Advanced"])

# ========== MODE 1: Data Analysis ==========
with tab1:
    uploaded_file = st.file_uploader("Upload file Excel/CSV untuk analisa data", type=["csv","xls","xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            xls = pd.ExcelFile(uploaded_file)
            df = pd.read_excel(uploaded_file, sheet_name=xls.sheet_names[0])
        st.subheader("Preview data")
        st.dataframe(df.head(10))
        categorical_cols, numeric_cols = detect_data_types(df)
        st.write(f"Kolom Numerik: {numeric_cols}")
        st.write(f"Kolom Kategorikal: {categorical_cols}")
        st.text(df_info_text(df))
        st.write(f"**Data shape:** {df.shape}")
        st.dataframe(safe_describe(df))

        if not df.select_dtypes(include="number").empty:
            st.write("**Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

# ========== MODE 2: RAG Advanced ==========
with tab2:
    uploaded_files = st.file_uploader(
        "Upload dokumen multi-file (PDF, TXT, DOCX, PPTX, CSV, XLSX, gambar)",
        type=["pdf","txt","docx","pptx","csv","xls","xlsx","png","jpg","jpeg","bmp"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.markdown("### 📂 Dokumen yang diupload:")
        for f in uploaded_files:
            st.write("- "+f.name)
        vectorstore = build_vectorstore(uploaded_files)
        retriever = vectorstore.as_retriever(search_kwargs={"k":3})
        st.success("✅ Dokumen siap diolah")

# ========== CHATBOT SELALU ==========
st.subheader("💬 Chatbot (Data Analysis + RAG)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Tanyakan sesuatu tentang data atau dokumen...")

if user_query:
    st.chat_message("user").markdown(user_query)
    llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-small", task="text2text-generation")
    # Gunakan RAG jika ada vectorstore
    if uploaded_files:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        result = qa_chain({"query": user_query})
        answer = result["result"]
        sources = result.get("source_documents",[])
        st.chat_message("assistant").markdown(answer)
        if sources:
            st.write("**Sumber:**")
            for s in sources:
                st.caption(f"{s.metadata.get('source','')} → {s.page_content[:200]}...")
    else:
        response = llm(user_query)
        st.chat_message("assistant").markdown(response)

    st.session_state.chat_history.append(("user",user_query))
    st.session_state.chat_history.append(("assistant",response if not uploaded_files else answer))

# Render history
for role,msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)
