# app-langchain.py

import streamlit as st
import os, io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import toml
from dotenv import load_dotenv, set_key

# LangChain & RAG
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline

# File loaders
from PyPDF2 import PdfReader
import docx
from pptx import Presentation
from PIL import Image
import requests
from transformers import pipeline

# ====== Load dotenv ======
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# ====== Sidebar: Google API Key ======
with st.sidebar:
    st.header("🔑 Konfigurasi API Key (Opsional)")
    GOOGLE_API_KEY = st.session_state.get("GOOGLE_API_KEY") or \
                     st.secrets.get("GOOGLE_API_KEY","") or \
                     os.getenv("GOOGLE_API_KEY","")

    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = st.text_input(
            "Buat GOOGLE API KEY baru pada https://aistudio.google.com/apikey kemudian copy dan paste disini",
            type="password"
        )
        if GOOGLE_API_KEY:
            st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            st.success("GOOGLE_API_KEY berhasil dimasukkan ✅")
            save_choice = st.radio("Simpan key ke mana?", ["Jangan simpan", ".env", "secrets.toml"])
            if st.button("💾 Simpan API Key"):
                if save_choice == ".env":
                    set_key(dotenv_path, "GOOGLE_API_KEY", GOOGLE_API_KEY)
                    st.success("✅ API Key disimpan ke .env")
                elif save_choice == "secrets.toml":
                    secrets_path = Path(".streamlit/secrets.toml")
                    secrets_path.parent.mkdir(exist_ok=True)
                    secrets = {}
                    if secrets_path.exists():
                        secrets = toml.load(secrets_path)
                    secrets["GOOGLE_API_KEY"] = GOOGLE_API_KEY
                    with open(secrets_path, "w") as f:
                        toml.dump(secrets, f)
                    st.success("✅ API Key disimpan ke .streamlit/secrets.toml")
    else:
        st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        st.success("GOOGLE_API_KEY berhasil dimuat ✅")

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

# ====== OCR.Space ======
def ocr_image(file):
    OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")
    if not OCR_SPACE_API_KEY:
        st.warning("⚠️ OCR_SPACE_API_KEY belum diset. Upload gambar tidak bisa di-OCR")
        return ""
    try:
        file.seek(0)
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": file},
            data={"apikey": OCR_SPACE_API_KEY, "language": "eng"}
        )
        data = resp.json()
        if data.get("ParsedResults"):
            text = data["ParsedResults"][0].get("ParsedText", "")
            return text
        return ""
    except Exception as e:
        st.warning(f"⚠️ OCR.Space error: {e}")
        return ""

# ====== Build Vectorstore ======
def build_vectorstore(files):
    docs = []
    for file in files:
        name = file.name
        try:
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
                text = "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape,"text")])
                docs.append(Document(page_content=text, metadata={"source": name}))
            elif name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                text = ocr_image(file)
                docs.append(Document(page_content=text, metadata={"source": name}))
            else:
                text = file.read().decode("utf-8", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": name}))
        except Exception as e:
            st.warning(f"⚠️ File {name} gagal diproses: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# ====== Main App ======
st.set_page_config(page_title="Data & Document RAG Chatbot", layout="wide")
st.title("📊🤖 Chatbot Analisis Data & Dokumen (RAG + HuggingFace Fallback)")

tab1, tab2 = st.tabs(["📊 Data Analysis", "📑 RAG Advanced"])

# ===== MODE 1: Data Analysis =====
with tab1:
    uploaded_file = st.file_uploader(
        "Upload file Excel/CSV untuk analisa data",
        type=["csv", "xls", "xlsx"]
    )
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state.dfs = {"CSV": df}
        else:
            xls = pd.ExcelFile(uploaded_file)
            st.session_state.dfs = {sheet: pd.read_excel(uploaded_file, sheet_name=sheet) for sheet in xls.sheet_names}

        st.subheader("📑 Pilih Sheet")
        sheet_names = list(st.session_state.dfs.keys())
        selected_sheets = st.multiselect("Sheet Aktif", sheet_names, default=sheet_names[:1])
        if not selected_sheets:
            st.warning("Pilih minimal satu sheet untuk analisis.")
            st.stop()

        if len(selected_sheets) == 1:
            df = st.session_state.dfs[selected_sheets[0]]
            sheet_label = selected_sheets[0]
        else:
            df_list = []
            for s in selected_sheets:
                temp = st.session_state.dfs[s].copy()
                temp["SheetName"] = s
                df_list.append(temp)
            df = pd.concat(df_list, ignore_index=True)
            sheet_label = ", ".join(selected_sheets)

        st.markdown(f"### 📄 Analisa: {uploaded_file.name} — Sheet(s): {sheet_label}")
        st.dataframe(df.head(10))
        categorical_cols, numeric_cols = detect_data_types(df)
        st.write(f"Kolom Numerik: {numeric_cols}")
        st.write(f"Kolom Kategorikal: {categorical_cols}")
        st.text(df_info_text(df))
        st.write(f"**Data shape:** {df.shape}")
        st.dataframe(safe_describe(df))

        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            st.write("**Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

# ===== MODE 2: RAG Advanced =====
with tab2:
    uploaded_files = st.file_uploader(
        "Upload dokumen (PDF, TXT, DOCX, PPTX, CSV, XLSX, gambar) → multi-file",
        type=["pdf","txt","docx","pptx","csv","xls","xlsx","png","jpg","jpeg","bmp"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.markdown("### 📂 Dokumen yang diupload:")
        for f in uploaded_files:
            st.write("- " + f.name)

        vectorstore = build_vectorstore(uploaded_files)
        retriever = vectorstore.as_retriever(search_kwargs={"k":3})

        # Gunakan HuggingFace pipeline sebagai LLM fallback
        hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", tokenizer="google/flan-t5-small")
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        st.success("✅ Chatbot RAG siap digunakan")

        if "chat_history_rag" not in st.session_state:
            st.session_state.chat_history_rag = []

        user_query = st.chat_input("Tanyakan sesuatu tentang dokumen...")
        if user_query:
            st.chat_message("user").markdown(user_query)
            st.session_state.chat_history_rag.append(("user", user_query))
            with st.spinner("🔎 Menganalisis dokumen..."):
                try:
                    result = qa_chain({"query": user_query})
                    answer = result["result"]
                    sources = result.get("source_documents", [])
                    st.chat_message("assistant").markdown(answer)
                    if sources:
                        st.write("**Sumber:**")
                        for s in sources:
                            st.caption(f"{s.metadata.get('source','')} → {s.page_content[:200]}...")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        for role, msg in st.session_state.chat_history_rag:
            st.chat_message(role).markdown(msg)
