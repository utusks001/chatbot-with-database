# app-multisheet-rag.py

import streamlit as st
import os, io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from dotenv import load_dotenv, set_key

# LangChain & RAG
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# PDF / OCR / Office
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import docx2txt
from pptx import Presentation

# ====== Load dotenv kalau lokal ======
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# ====== Helper deteksi local vs Streamlit Cloud ======
def is_streamlit_cloud():
    return os.environ.get("STREAMLIT_RUNTIME") is not None

# ====== API Key Management ======
with st.sidebar:
    st.header("🔑 Konfigurasi API Key")

    GOOGLE_API_KEY = (
        st.secrets.get("GOOGLE_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
    )

    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = st.text_input(
            "Masukkan GOOGLE_API_KEY (buat di https://aistudio.google.com/apikey)",
            type="password"
        )
        if GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            st.success("✅ GOOGLE_API_KEY berhasil dimasukkan")

            if is_streamlit_cloud():
                st.info("ℹ️ Running di Streamlit Cloud → gunakan menu Settings → Secrets untuk simpan permanen")
            else:
                # Save ke .env lokal
                set_key(dotenv_path, "GOOGLE_API_KEY", GOOGLE_API_KEY)
                st.success("✅ API Key disimpan ke .env (lokal)")
    else:
        st.success("✅ GOOGLE_API_KEY berhasil dimuat")

    # Embeddings toggle
    st.subheader("⚙️ Embeddings Settings")
    use_hf_embeddings = st.checkbox("Pakai HuggingFace embeddings saja", value=False)
    st.session_state["USE_HF_EMBEDDINGS"] = use_hf_embeddings

# ====== Helper untuk Data Analysis ======
def safe_describe(df: pd.DataFrame):
    try:
        return df.describe(include="all").transpose()
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def df_info_text(df: pd.DataFrame):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

# ====== Helper RAG Vectorstore ======
def build_vectorstore(files):
    """Bangun vectorstore dari multi dokumen (txt, pdf, csv, xlsx, docx, pptx, gambar OCR)."""
    docs = []

    for file in files:
        name = file.name.lower()

        if name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": file.name}))

        elif name.endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            docs.append(Document(page_content=text, metadata={"source": file.name}))

        elif name.endswith(".csv"):
            df = pd.read_csv(file)
            docs.append(Document(page_content=df.to_csv(index=False), metadata={"source": file.name}))

        elif name.endswith((".xls", ".xlsx")):
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet)
                text = df.to_csv(index=False)
                docs.append(Document(page_content=text, metadata={"source": f"{file.name}#{sheet}"}))

        elif name.endswith(".docx"):
            text = docx2txt.process(file)
            docs.append(Document(page_content=text, metadata={"source": file.name}))

        elif name.endswith(".pptx"):
            prs = Presentation(file)
            text = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text.append(shape.text)
            docs.append(Document(page_content="\n".join(text), metadata={"source": file.name}))

        elif name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = Image.open(file)
            text = pytesseract.image_to_string(image, lang="eng+ind")
            docs.append(Document(page_content=text, metadata={"source": file.name}))

        else:
            try:
                text = file.read().decode("utf-8", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": file.name}))
            except Exception:
                st.warning(f"⚠️ Format file {file.name} tidak dikenali")

    # Split dokumen
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # ====== Embeddings Selection ======
    try:
        if st.session_state.get("USE_HF_EMBEDDINGS", False):
            raise RuntimeError("Manual HF mode dipilih")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY", "")
        )
        return FAISS.from_documents(split_docs, embeddings)

    except Exception as e:
        st.warning(f"⚠️ Gagal pakai Gemini embeddings ({e}). Fallback ke HuggingFace & auto-toggle.")
        st.session_state["USE_HF_EMBEDDINGS"] = True
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(split_docs, embeddings)

# ====== Main App ======
st.set_page_config(page_title="Data & Document Chatbot", layout="wide")
st.title("🤖 Chatbot Analisis Data & Dokumen (Gemini/HuggingFace)")

menu = st.radio("Pilih mode:", ["📊 Data Analysis", "📑 Advanced RAG"])

# ====== Mode 1: Data Analysis (Multi-sheet Excel/CSV) ======
if menu == "📊 Data Analysis":
    uploaded_file = st.file_uploader(
        "Upload file Excel (.xls, .xlsx) atau CSV (.csv)", 
        type=["csv", "xls", "xlsx"]
    )

    if uploaded_file is not None:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.session_state.dfs = {"CSV": df}
        else:
            xls = pd.ExcelFile(uploaded_file)
            st.session_state.dfs = {sheet: pd.read_excel(uploaded_file, sheet_name=sheet) for sheet in xls.sheet_names}

        # Multi-select sheet
        sheet_names = list(st.session_state.dfs.keys())
        selected_sheets = st.multiselect("📑 Pilih Sheet", sheet_names, default=sheet_names[:1])

        if not selected_sheets:
            st.warning("Pilih minimal satu sheet untuk analisis.")
            st.stop()

        # Gabung data
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
        st.write("**Info():**")
        st.text(df_info_text(df))
        st.write("**Describe():**")
        st.dataframe(safe_describe(df))

        # Heatmap
        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            st.write("**Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # Agent LLM untuk analisis
        agent_key = f"agent_{sheet_label}"
        if agent_key not in st.session_state and os.getenv("GOOGLE_API_KEY"):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                st.session_state[agent_key] = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code=True
                )
                st.success(f"✅ Chatbot siap! (Sheet: {sheet_label})")
            except Exception as e:
                st.error(f"Gagal inisialisasi chatbot: {e}")
                st.stop()

        if os.getenv("GOOGLE_API_KEY"):
            user_query = st.chat_input("Tanyakan sesuatu tentang data...")
            if user_query:
                with st.chat_message("user"):
                    st.markdown(user_query)
                with st.spinner("Memproses..."):
                    try:
                        response = st.session_state[agent_key].run(user_query)
                        with st.chat_message("assistant"):
                            st.markdown(response)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ====== Mode 2: Advanced RAG (Multi-file, OCR, QA) ======
elif menu == "📑 Advanced RAG":
    uploaded_files = st.file_uploader(
        "Upload dokumen (PDF, TXT, DOCX, PPTX, CSV, XLSX, Gambar: JPG, PNG, BMP, dsb)", 
        type=["csv", "xls", "xlsx", "pdf", "txt", "docx", "pptx", "png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown("### 📂 Dokumen terunggah:")
        for f in uploaded_files:
            st.write("•", f.name)

        vectorstore = build_vectorstore(uploaded_files)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        if "qa_chain" not in st.session_state:
            if os.getenv("GOOGLE_API_KEY"):
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
            else:
                st.stop()

            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            st.success("✅ Chatbot RAG siap digunakan")

        user_query = st.chat_input("Tanyakan sesuatu tentang dokumen hukum/medis yang diupload...")
        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.spinner("🔎 Menganalisis dokumen..."):
                try:
                    result = st.session_state.qa_chain({"query": user_query})
                    answer = result["result"]
                    sources = result.get("source_documents", [])

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if sources:
                            st.write("**Sumber:**")
                            for s in sources:
                                st.caption(f"[{s.metadata.get('source','')}] {s.page_content[:200]}...")
                except Exception as e:
                    st.error(f"❌ Error saat menjawab: {e}")
