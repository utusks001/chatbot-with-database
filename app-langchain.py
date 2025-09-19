# app-langchain.py

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredImageLoader
)
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ========== LLM LOADER ==========
def load_llm():
    if os.getenv("GEMINI_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    elif os.getenv("GROQ_API_KEY"):
        return ChatGroq(model="llama3-8b-8192", temperature=0)
    else:
        st.warning("⚠️ No LLM API key found. Using fallback mode.")
        return None

llm = load_llm()

def detect_data_types(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    return cat_cols, num_cols

def df_info_text(df: pd.DataFrame) -> str:
    """Ringkasan dataset sederhana"""
    info = f"Baris: {df.shape[0]}, Kolom: {df.shape[1]}\n"
    info += "Kolom:\n" + ", ".join(df.columns[:10])
    if df.shape[1] > 10:
        info += " ..."
    return info

def safe_describe(df):
    try:
        return df.describe(include="all")
    except Exception:
        return pd.DataFrame()

# ========== Streamlit CONFIG ==========
st.set_page_config(page_title="📊 Data Analysis + 📚 RAG Chatbot", layout="wide")
st.title("🤖 Chatbot: Data Analysis + RAG Advanced")

tab1, tab2 = st.tabs(["📈 Data Analysis", "📚 RAG Advanced"])

# ========== TAB 1: DATA ANALYSIS ==========
with tab1:
    st.subheader("Upload Dataset")
    file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.dataframe(df.head(10))
        categorical_cols, numeric_cols = detect_data_types(df)
        st.write(f"Kolom Numerik: {numeric_cols}")
        st.write(f"Kolom Kategorikal: {categorical_cols}")
        st.text(df_info_text(df))
        st.write(f"**Data shape:** {df.shape}")
        
        # Dropdown for visualization
        x_axis = st.selectbox("Pilih X-Axis", df.columns, index=0)
        y_axis = st.selectbox("Pilih Y-Axis", df.columns, index=min(1, len(df.columns)-1))

        q = st.text_input("💬 Tanya tentang dataset:")
        if q:
            if "trend" in q.lower():
                fig = px.line(df, x=x_axis, y=y_axis, title=f"Trend {y_axis} vs {x_axis}")
                st.plotly_chart(fig, use_container_width=True)

                # Insight with LLM if available
                if llm:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "Kamu adalah analis data yang ringkas."),
                        ("human", f"Buat insight ringkas dari tren {y_axis} terhadap {x_axis} pada dataset ini:\n{df[[x_axis,y_axis]].head(20)}")
                    ])
                    response = llm.invoke(prompt.format_messages())
                    st.success(response.content)
                else:
                    st.info(f"ℹ️ {y_axis} meningkat/menurun terhadap {x_axis} berdasarkan grafik.")
            elif "statistik" in q.lower() or "summary" in q.lower():
                st.dataframe(df.describe(include="all"))
            elif "insight" in q.lower() or "kesimpulan" in q.lower():
                if llm:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "Kamu adalah analis data."),
                        ("human", f"Buat insight & kesimpulan utama dari dataset ini:\n{df.head(30)}")
                    ])
                    response = llm.invoke(prompt.format_messages())
                    st.success(response.content)
                else:
                    st.info("Dataset menunjukkan pola tertentu, silakan analisis lebih lanjut.")
            else:
                st.write("📊 Statistik umum:")
                st.dataframe(df.describe(include="all"))

# ========== TAB 2: RAG ADVANCED ==========
with tab2:
    st.subheader("Upload Dokumen (TXT, PDF, DOCX, PPTX, Gambar)")
    rag_file = st.file_uploader("Upload file untuk RAG", type=["txt", "pdf", "docx", "pptx", "jpg", "png", "bmp", "gif", "tiff"])

    if rag_file:
        file_path = os.path.join("temp", rag_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(rag_file.read())

        # Loader sesuai tipe file
        if rag_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif rag_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif rag_file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(file_path)
        elif rag_file.name.endswith((".jpg",".png",".bmp",".gif",".tiff")):
            loader = UnstructuredImageLoader(file_path)
        else:
            loader = TextLoader(file_path)

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        q2 = st.text_input("💬 Tanya tentang dokumen:")
        if q2:
            results = vectorstore.similarity_search(q2, k=3)
            context = "\n\n".join([r.page_content for r in results])

            if llm:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Kamu adalah asisten RAG yang menjawab berdasarkan dokumen."),
                    ("human", f"Pertanyaan: {q2}\n\nKonteks:\n{context}")
                ])
                response = llm.invoke(prompt.format_messages())
                st.success(response.content)
            else:
                st.info("⚠️ Tidak ada LLM API key. Jawaban tidak tersedia.")
