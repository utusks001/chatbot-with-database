# app-multisheet-rag.py

import streamlit as st
import os, io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import toml
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

# Document loaders
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
from PIL import Image
from paddleocr import PaddleOCR

# Load dotenv kalau lokal
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

# ====== Build Vectorstore ======
def build_vectorstore(files):
    """Bangun vectorstore dari multi-file upload."""
    docs = []
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    for file in files:
        if file.name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": file.name}))

        elif file.name.endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            docs.append(Document(page_content=text, metadata={"source": file.name}))

        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            text = df.to_csv(index=False)
            docs.append(Document(page_content=text, metadata={"source": file.name}))

        elif file.name.endswith((".xls", ".xlsx")):
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                text = pd.read_excel(file, sheet_name=sheet).to_csv(index=False)
                docs.append(Document(page_content=f"[{sheet}]\n{text}", metadata={"source": f"{file.name}:{sheet}"}))

        elif file.name.endswith(".docx"):
            doc = DocxDocument(file)
            text = "\n".join([p.text for p in doc.paragraphs])
            docs.append(Document(page_content=text, metadata={"source": file.name}))

        elif file.name.endswith(".pptx"):
            prs = Presentation(file)
            text = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text.append(shape.text)
            docs.append(Document(page_content="\n".join(text), metadata={"source": file.name}))

        elif file.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image = Image.open(file)
            result = ocr.ocr(file.name, cls=True)
            text = " ".join([line[1][0] for page in result for line in page])
            docs.append(Document(page_content=text, metadata={"source": file.name}))

        else:
            try:
                text = file.read().decode("utf-8", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": file.name}))
            except:
                pass

    # Split dokumen
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # ====== Embeddings Selection ======
    if st.session_state.get("USE_HF_EMBEDDINGS", False):
        st.info("ℹ️ Mode HuggingFace embeddings dipilih (manual).")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(split_docs, embeddings)

    # Default: coba Gemini embeddings → fallback HuggingFace
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY", "")
        )
        return FAISS.from_documents(split_docs, embeddings)
    except Exception as e:
        st.warning(f"⚠️ Gagal pakai Gemini embeddings ({e}). Fallback ke HuggingFace.")
        st.session_state["USE_HF_EMBEDDINGS"] = True
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(split_docs, embeddings)

# ====== Main App ======
st.set_page_config(page_title="Data & Document RAG Chatbot", layout="wide")
st.title("📑🤖 Chatbot Analisis Data & Dokumen (RAG + Gemini/HuggingFace)")

menu = st.radio("Pilih Mode:", ["Data Analysis (Excel/CSV)", "Advanced RAG (Multi-file Docs)"])

# ====== Data Analysis ======
if menu == "Data Analysis (Excel/CSV)":
    uploaded_file = st.file_uploader("Upload file Excel / CSV", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(("xls", "xlsx")) else pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        st.write("### Info Data")
        st.text(df_info_text(df))

        st.write("### Deskripsi Statistik")
        st.dataframe(safe_describe(df))

        # Visualisasi
        st.write("### Visualisasi Data")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            col = st.selectbox("Pilih kolom numerik", num_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.info("Tidak ada kolom numerik untuk divisualisasikan.")

        # Chatbot analisa data
        if "agent" not in st.session_state and os.getenv("GOOGLE_API_KEY"):
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            st.session_state.agent = create_pandas_dataframe_agent(
                llm, df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
            )
            st.success("✅ Chatbot Data Analysis siap!")

        user_query = st.chat_input("Tanyakan sesuatu tentang data...")
        if user_query and "agent" in st.session_state:
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.spinner("🔎 Menganalisis data..."):
                try:
                    response = st.session_state.agent.run(user_query)
                    with st.chat_message("assistant"):
                        st.markdown(response)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ====== Advanced RAG ======
if menu == "Advanced RAG (Multi-file Docs)":
    uploaded_files = st.file_uploader(
        "Upload dokumen (PDF, TXT, DOCX, PPTX, CSV, XLSX, JPG, PNG, BMP)",
        type=["csv", "xls", "xlsx", "pdf", "txt", "docx", "pptx", "jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"### 📂 {len(uploaded_files)} file diupload")

        # Bangun vectorstore
        vectorstore = build_vectorstore(uploaded_files)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

        # Chat input
        user_query = st.chat_input("Tanyakan sesuatu tentang dokumen hukum/medis...")
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
                                st.caption(f"{s.metadata.get('source', '')}: {s.page_content[:200]}...")
                except Exception as e:
                    st.error(f"❌ Error saat menjawab: {e}")
