# app-langchain.py

import streamlit as st
import pandas as pd
import plotly.express as px
import fitz  # PyMuPDF
import docx
from pptx import Presentation
import pytesseract
from PIL import Image
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Chatbot Data Analysis + RAG", layout="wide")


# ========== HELPERS ==========
def safe_describe(df: pd.DataFrame):
    try:
        return df.describe(include="all").transpose()
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def detect_date_columns(df: pd.DataFrame):
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if df[col].notna().sum() > 0:
                    date_cols.append(col)
            except:
                pass
    return date_cols


def detect_categorical_columns(df: pd.DataFrame):
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            cat_cols.append(col)
        elif df[col].nunique() < 20:
            cat_cols.append(col)
    return cat_cols


# Extract text from different file types
def extract_text_from_file(uploaded_file):
    text = ""
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in pdf_doc:
            text += page.get_text()

    elif filename.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif filename.endswith(".pptx"):
        prs = Presentation(uploaded_file)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"

    elif filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")):
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)

    return text


# Simple RAG using TF-IDF
def rag_answer(query, documents):
    if not documents:
        return "⚠️ Tidak ada dokumen diunggah."

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_vectors).flatten()
    best_idx = np.argmax(sims)
    return f"📄 Jawaban berdasarkan dokumen:\n\n{documents[best_idx][:1000]}"


# ========== APP ==========
st.title("📊 Chatbot Data Analysis + 📑 RAG Advanced")

tab1, tab2 = st.tabs(["📊 Data Analysis", "📑 RAG Advanced"])


# ================= TAB 1: DATA ANALYSIS =================
with tab1:
    st.header("Upload Dataset (CSV/Excel)")
    uploaded_file = st.file_uploader("Unggah dataset Anda", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ Dataset berhasil dimuat! {df.shape[0]} baris, {df.shape[1]} kolom")
        st.dataframe(df.head())

        if "analysis_chat" not in st.session_state:
            st.session_state.analysis_chat = []

        st.subheader("💬 Chatbot Data Analysis")

        for msg in st.session_state.analysis_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Tanyakan tentang dataset (contoh: statistik, trend sales, grafik kategori, insight)")
        if user_query:
            st.session_state.analysis_chat.append({"role": "user", "content": user_query})
            response = "⚠️ Saya belum mengerti pertanyaan Anda."

            # === Statistik ===
            if "statistik" in user_query.lower():
                stats = safe_describe(df)
                st.dataframe(stats)
                response = f"📊 Dataset: {df.shape[0]} baris, {df.shape[1]} kolom."
                if "Sales" in df.columns:
                    sales_summary = df["Sales"].describe()
                    response += (
                        f"\n- Rata-rata Sales: {sales_summary['mean']:.2f}"
                        f"\n- Min Sales: {sales_summary['min']:.2f}"
                        f"\n- Max Sales: {sales_summary['max']:.2f}"
                    )

            # === Trend ===
            elif "trend" in user_query.lower():
                date_cols = detect_date_columns(df)
                if date_cols and "Sales" in df.columns:
                    date_col = date_cols[0]
                    trend_df = df.groupby(date_col)["Sales"].sum().reset_index()
                    fig = px.line(trend_df, x=date_col, y="Sales", title="📈 Trend Sales")
                    st.plotly_chart(fig, use_container_width=True)
                    response = f"📈 Trend Sales ditampilkan berdasarkan kolom `{date_col}`."
                else:
                    response = "⚠️ Tidak ada kolom tanggal atau kolom Sales."

            # === Kategori ===
            elif "kategori" in user_query.lower() or "grafik" in user_query.lower():
                cat_cols = detect_categorical_columns(df)
                if cat_cols and "Sales" in df.columns:
                    col_choice = st.selectbox("Pilih kolom kategori:", cat_cols)
                    if col_choice:
                        cat_df = df.groupby(col_choice)["Sales"].sum().reset_index()
                        fig = px.bar(cat_df, x=col_choice, y="Sales", title=f"📊 Sales per {col_choice}")
                        st.plotly_chart(fig, use_container_width=True)
                        response = f"📊 Grafik Sales per kategori `{col_choice}` ditampilkan."
                else:
                    response = "⚠️ Tidak ada kolom kategori cocok."

            # === Insight ===
            elif "insight" in user_query.lower():
                response = f"🔎 Insight singkat:\n- Top kategori: {df.select_dtypes(include='object').nunique().idxmax()}\n- Nilai sales total: {df['Sales'].sum() if 'Sales' in df.columns else 'N/A'}"

            # === Kesimpulan ===
            elif "kesimpulan" in user_query.lower() or "ringkas" in user_query.lower():
                response = f"📌 Kesimpulan dataset: {df.shape[0]} baris, {df.shape[1]} kolom. Data cukup besar untuk analisis lanjutan."

            st.session_state.analysis_chat.append({"role": "assistant", "content": response})


# ================= TAB 2: RAG ADVANCED =================
with tab2:
    st.header("Upload Dokumen untuk RAG Advanced")
    rag_file = st.file_uploader(
        "Unggah dokumen (TXT, PDF, DOCX, PPTX, JPG, PNG, BMP, TIFF, GIF)",
        type=["txt", "pdf", "docx", "pptx", "jpg", "jpeg", "png", "bmp", "tiff", "gif"],
        key="rag"
    )

    if rag_file:
        text_content = extract_text_from_file(rag_file)
        if "rag_docs" not in st.session_state:
            st.session_state.rag_docs = []
        st.session_state.rag_docs.append(text_content)

        st.success(f"✅ Dokumen '{rag_file.name}' berhasil diunggah & diproses.")

        if "rag_chat" not in st.session_state:
            st.session_state.rag_chat = []

        st.subheader("💬 Chatbot RAG Advanced")

        for msg in st.session_state.rag_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        rag_query = st.chat_input("Tanyakan isi dokumen (contoh: ringkas dokumen, cari topik penting)", key="rag_chat_input")
        if rag_query:
            st.session_state.rag_chat.append({"role": "user", "content": rag_query})
            response = rag_answer(rag_query, st.session_state.rag_docs)
            st.session_state.rag_chat.append({"role": "assistant", "content": response})
