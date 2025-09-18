# app-langchain.py

# =======================
# app.py - Ultra-lite RAG + Data Analysis
# Compatible Python 3.11 + Streamlit Cloud
# =======================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import requests
from PIL import Image
import toml

# File loaders
from PyPDF2 import PdfReader
import docx
from pptx import Presentation

# LangChain & RAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# =======================
# Sidebar: Manual API key (optional, placeholder)
# =======================
st.sidebar.header("🔑 API Configuration")
st.sidebar.info("This app uses HuggingFace fallback; no API key needed.")

# =======================
# Helper Functions
# =======================
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

# =======================
# OCR.Space helper
# =======================
OCR_SPACE_API_KEY = ""  # Optional: set your key in .env if you want

def ocr_image(file):
    file.seek(0)
    if OCR_SPACE_API_KEY:
        try:
            resp = requests.post(
                "https://api.ocr.space/parse/image",
                files={"file": file},
                data={"apikey": OCR_SPACE_API_KEY, "language": "eng"}
            )
            data = resp.json()
            if data.get("ParsedResults"):
                return data["ParsedResults"][0].get("ParsedText", "")
        except:
            return ""
    return ""

# =======================
# Build Vectorstore (multi-file)
# =======================
def build_vectorstore(files):
    docs = []
    for file in files:
        name = file.name

        if name.endswith(".txt"):
            text = file.read().decode("utf-8")
            docs.append(Document(page_content=text, metadata={"source": name}))

        elif name.endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            docs.append(Document(page_content=text, metadata={"source": name}))

        elif name.endswith(".csv"):
            df = pd.read_csv(file)
            text = df.to_csv(index=False)
            docs.append(Document(page_content=text, metadata={"source": name}))

        elif name.endswith(".xlsx") or name.endswith(".xls"):
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet)
                text = f"[{sheet}]\n" + df.to_csv(index=False)
                docs.append(Document(page_content=text, metadata={"source": f"{name}:{sheet}"}))

        elif name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            docs.append(Document(page_content=text, metadata={"source": name}))

        elif name.endswith(".pptx"):
            prs = Presentation(file)
            text = "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])
            docs.append(Document(page_content=text, metadata={"source": name}))

        elif name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            text = ocr_image(file)
            docs.append(Document(page_content=text, metadata={"source": name}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(split_docs, embeddings)

# =======================
# Main App
# =======================
st.set_page_config(page_title="Ultra-lite Data & RAG Chatbot", layout="wide")
st.title("📊🤖 Chatbot Data Analysis + RAG (Ultra-lite HuggingFace)")

tab1, tab2 = st.tabs(["📊 Data Analysis", "📑 RAG Advanced"])

# =======================
# Tab 1: Data Analysis
# =======================
with tab1:
    uploaded_file = st.file_uploader("Upload Excel/CSV for Analysis", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state.dfs = {"CSV": df}
        else:
            xls = pd.ExcelFile(uploaded_file)
            st.session_state.dfs = {sheet: pd.read_excel(uploaded_file, sheet_name=sheet) for sheet in xls.sheet_names}

        sheet_names = list(st.session_state.dfs.keys())
        selected_sheets = st.multiselect("Select Sheets", sheet_names, default=sheet_names[:1])
        if not selected_sheets:
            st.warning("Select at least one sheet.")
            st.stop()

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
        st.write(f"Numeric columns: {numeric_cols}")
        st.write(f"Categorical columns: {categorical_cols}")
        st.dataframe(safe_describe(df))

        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # =======================
        # Data Analysis Chatbot (HuggingFace fallback)
        # =======================
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_query = st.chat_input("Ask about the dataset...")
        if user_query:
            st.chat_message("user").markdown(user_query)
            st.session_state.chat_history.append(("user", user_query))

            # Simple HuggingFace QA pipeline
            try:
                hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
                preview = df.head(1000).to_csv(index=False)
                prompt = f"Provide key insights and conclusion from the following dataset:\n{preview}\nQuestion: {user_query}"
                result = hf_pipeline(prompt, max_length=300)[0]["generated_text"]
                st.chat_message("assistant").markdown(result)
                st.session_state.chat_history.append(("assistant", result))
            except Exception as e:
                st.error(f"Error: {e}")

        for role, msg in st.session_state.chat_history:
            st.chat_message(role).markdown(msg)

# =======================
# Tab 2: RAG Advanced
# =======================
with tab2:
    uploaded_files = st.file_uploader("Upload multi-file for RAG (PDF, TXT, DOCX, PPTX, CSV, XLSX, Images)", type=["pdf","txt","docx","pptx","csv","xls","xlsx","png","jpg","jpeg","bmp"], accept_multiple_files=True)
    if uploaded_files:
        st.markdown("### Uploaded Files")
        for f in uploaded_files:
            st.write(f"- {f.name}")

        vectorstore = build_vectorstore(uploaded_files)
        retriever = vectorstore.as_retriever(search_kwargs={"k":3})

        if "rag_chat_history" not in st.session_state:
            st.session_state.rag_chat_history = []

        user_query = st.chat_input("Ask something about the documents...")
        if user_query:
            st.chat_message("user").markdown(user_query)
            st.session_state.rag_chat_history.append(("user", user_query))

            try:
                hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
                # Build context from top-k docs
                docs = retriever.get_relevant_documents(user_query)
                context = "\n".join([d.page_content[:500] for d in docs])
                prompt = f"Answer the question based on the following context:\n{context}\nQuestion: {user_query}"
                answer = hf_pipeline(prompt, max_length=300)[0]["generated_text"]
                st.chat_message("assistant").markdown(answer)
                st.session_state.rag_chat_history.append(("assistant", answer))
            except Exception as e:
                st.error(f"Error: {e}")

        for role, msg in st.session_state.rag_chat_history:
            st.chat_message(role).markdown(msg)
