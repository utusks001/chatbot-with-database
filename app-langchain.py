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
                text = file.read().decode("utf-8", errors="ignore")
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

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(split_docs, embeddings)

# ====== Main App ======
st.set_page_config(page_title="Data & Document RAG Chatbot", layout="wide")
st.title("📊🤖 Chatbot Analisis Data & Dokumen (RAG + HF)")

# ====== Session State Init ======
for key in ["dfs", "uploaded_file", "uploaded_files", "chat_history_data", "chat_history_rag", "vectorstore", "active_tab"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "chat_history" in key else None

# ====== Tabs ======
tab1, tab2 = st.tabs(["📊 Data Analysis", "📑 RAG Advanced"])

with tab1:
    st.session_state.active_tab = "Data Analysis"
    uploaded_file = st.file_uploader("Upload file Excel/CSV untuk analisa data", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    df = None
    if st.session_state.uploaded_file:
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

with tab2:
    st.session_state.active_tab = "RAG Advanced"
    uploaded_files = st.file_uploader(
        "Upload dokumen (PDF, TXT, DOCX, PPTX, CSV, XLSX, gambar) → bisa multi-file",
        type=["pdf", "txt", "docx", "pptx", "csv", "xls", "xlsx", "png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.vectorstore = build_vectorstore(uploaded_files)
        if st.session_state.vectorstore:
            st.success("✅ Dokumen berhasil diproses")

# ====== LLM Provider ======
def get_llm():
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                          google_api_key=os.getenv("GOOGLE_API_KEY"),
                                          temperature=0.2)
        except:
            pass
    try:
        return HuggingFacePipeline.from_model_id(model_id="google/flan-t5-small", task="text2text-generation")
    except:
        return None

# ====== CHATBOT ROOT-LEVEL ======
user_query = st.chat_input("💬 Tanyakan sesuatu...")

if st.session_state.active_tab == "Data Analysis":
    if "chat_history_data" not in st.session_state:
        st.session_state.chat_history_data = []
    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history_data.append(("user", user_query))
        llm = get_llm()
        if llm:
            df = None
            if st.session_state.dfs:
                df_list = list(st.session_state.dfs.values())
                df = pd.concat(df_list, ignore_index=True)
            preview = df.head(1000).to_csv(index=False) if df is not None else ""
            prompt = f"Anda adalah asisten analisis data.\nDataset sampel:\n{preview}\nPertanyaan: {user_query}"
            try:
                response = llm.invoke(prompt).content
            except:
                response = "⚠️ LLM tidak tersedia."
        else:
            response = "⚠️ Tidak ada LLM yang tersedia."
        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history_data.append(("assistant", response))
    for role, msg in st.session_state.chat_history_data:
        st.chat_message(role).markdown(msg)

elif st.session_state.active_tab == "RAG Advanced":
    if "chat_history_rag" not in st.session_state:
        st.session_state.chat_history_rag = []
    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history_rag.append(("user", user_query))
        llm = get_llm()
        if llm and st.session_state.vectorstore:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            try:
                result = qa_chain({"query": user_query})
                answer = result["result"]
                st.chat_message("assistant").markdown(answer)
                st.session_state.chat_history_rag.append(("assistant", answer))
                sources = result.get("source_documents", [])
                if sources:
                    st.write("**Sumber:**")
                    for s in sources:
                        st.caption(f"{s.metadata.get('source','')} → {s.page_content[:200]}...")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            response = "⚠️ Belum ada dokumen/LLM."
            st.chat_message("assistant").markdown(response)
            st.session_state.chat_history_rag.append(("assistant", response))
    for role, msg in st.session_state.chat_history_rag:
        st.chat_message(role).markdown(msg)
