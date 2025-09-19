# app-langchain.py

import os
import tempfile
import pandas as pd
import streamlit as st
import plotly.express as px

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    UnstructuredImageLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ======================
# LLM SETUP (Gemini + Groq fallback)
# ======================
def load_llm():
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    except Exception:
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

llm = load_llm()

# ======================
# Helpers
# ======================
def df_info_text(df: pd.DataFrame) -> str:
    """Ringkasan dataset dalam teks."""
    buf = []
    buf.append(f"Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom.")
    buf.append("Kolom: " + ", ".join(df.columns))
    buf.append("5 baris pertama:\n" + str(df.head().to_dict(orient="records")))
    return "\n".join(buf)

def generate_insight(df: pd.DataFrame):
    """Gunakan LLM untuk insight natural."""
    template = """
    Kamu adalah asisten data analyst. Buat insight singkat dan kesimpulan dari dataset berikut:

    {info}

    Jawab dengan ringkas dan natural.
    """
    prompt = PromptTemplate(input_variables=["info"], template=template)
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain.run(info=df_info_text(df))

def build_rag_qa(uploaded_files):
    """RAG pipeline untuk dokumen + gambar"""
    docs = []
    for file in uploaded_files:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in ["docx"]:
            loader = Docx2txtLoader(tmp_path)
        elif suffix in ["pptx"]:
            loader = UnstructuredPowerPointLoader(tmp_path)
        elif suffix in ["txt", "csv", "md"]:
            loader = TextLoader(tmp_path)
        elif suffix in ["jpg", "jpeg", "png", "bmp", "jiff", "gif"]:
            loader = UnstructuredImageLoader(tmp_path)
        else:
            continue

        docs.extend(loader.load())

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="📊 Data + RAG Chatbot", layout="wide")
st.title("📊 Data Analysis + 📑 RAG Chatbot")

tab1, tab2 = st.tabs(["📈 Data Analysis", "📑 RAG Advanced"])

# ======================
# TAB 1: Data Analysis
# ======================
with tab1:
    uploaded_file = st.file_uploader("Unggah dataset (CSV atau Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("Dataset berhasil dimuat ✅")
            st.dataframe(df.head())

            # Pilihan X/Y untuk grafik
            st.subheader("⚙️ Pilih Kolom untuk Visualisasi")
            x_axis = st.selectbox("Kolom X Axis", df.columns, index=0)
            y_axis = st.selectbox("Kolom Y Axis", df.columns, index=min(1, len(df.columns)-1))

            # Tampilkan grafik otomatis
            if x_axis and y_axis:
                fig = px.line(df, x=x_axis, y=y_axis, title=f"📈 Grafik {y_axis} vs {x_axis}")
                st.plotly_chart(fig, use_container_width=True)

            # Chatbot Analysis
            st.subheader("💬 Chatbot Data Analysis")
            q = st.text_input("Ajukan pertanyaan tentang dataset:")
            if q:
                if any(k in q.lower() for k in ["statistik", "tren", "kategori", "ringkas", "insight", "kesimpulan"]):
                    try:
                        ans = generate_insight(df)
                        st.info(ans)
                    except Exception:
                        st.warning("⚠️ Gagal membuat insight otomatis.")
                else:
                    try:
                        result = llm.invoke(q + "\nGunakan dataset berikut:\n" + df_info_text(df))
                        st.info(result.content if hasattr(result, "content") else result)
                    except Exception as e:
                        st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Gagal membaca dataset: {e}")

# ======================
# TAB 2: RAG Advanced
# ======================
with tab2:
    uploaded_docs = st.file_uploader(
        "Unggah dokumen (TXT, PDF, DOCX, PPTX, JPG, PNG, BMP, JIFF, GIF)", 
        type=["txt", "pdf", "docx", "pptx", "jpg", "jpeg", "png", "bmp", "jiff", "gif"], 
        accept_multiple_files=True
    )
    if uploaded_docs:
        qa = build_rag_qa(uploaded_docs)
        if qa:
            st.success("Dokumen berhasil diproses ✅")
            query = st.text_input("Tanyakan sesuatu dari dokumen:")
            if query:
                try:
                    ans = qa.run(query)
                    st.info(ans)
                except Exception as e:
                    st.error(f"Error QA: {e}")
        else:
            st.warning("❌ Tidak ada dokumen yang valid.")
