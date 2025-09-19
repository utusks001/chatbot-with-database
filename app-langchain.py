# app-langchain.py

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

from PyPDF2 import PdfReader
from pathlib import Path
import docx
from pptx import Presentation
from PIL import Image
import requests

# LangChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

st.set_page_config(page_title="Ultra-lite Data Analysis + RAG", layout="wide")

# ================= Session State Init =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "active_df" not in st.session_state:
    st.session_state.active_df = None

# ================= Helper =================
def safe_read_file(file):
    """Ekstrak teks dari berbagai format"""
    name = file.name
    text = ""
    try:
        if name.endswith(".txt"):
            text = file.read().decode("utf-8")
        elif name.endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif name.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(file)
            text = ""
            for sheet in xls.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet)
                text += f"[{sheet}]\n" + df.to_csv(index=False) + "\n"
        elif name.endswith(".csv"):
            df = pd.read_csv(file)
            text = df.to_csv(index=False)
        elif name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif name.endswith(".pptx"):
            prs = Presentation(file)
            text = "\n".join(
                [shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")]
            )
        elif name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            api_key = st.secrets.get("OCR_SPACE_API_KEY", "")
            if api_key:
                file.seek(0)
                resp = requests.post(
                    "https://api.ocr.space/parse/image",
                    files={"file": file},
                    data={"apikey": api_key, "language": "eng"}
                )
                data = resp.json()
                if "ParsedResults" in data and data["ParsedResults"]:
                    text = data["ParsedResults"][0].get("ParsedText", "")
        else:
            text = file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"⚠️ Gagal baca {name}: {e}")
    return text

def build_vectorstore(files):
    docs = []
    for f in files:
        f.seek(0)
        text = safe_read_file(f)
        if text:
            docs.append(Document(page_content=text, metadata={"source": f.name}))
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(split_docs, embeddings)

def detect_data_types(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    return cat_cols, num_cols

def safe_describe(df):
    try:
        return df.describe(include="all")
    except Exception:
        return pd.DataFrame()

def qa_general_dataset(df, query: str):
    query_lower = query.lower()
    cols = df.columns.str.lower()
    match_cols = [c for c in cols if c in query_lower]

    if "rata" in query_lower or "average" in query_lower or "mean" in query_lower:
        if match_cols:
            col = df.columns[cols.get_loc(match_cols[0])]
            return f"📊 Rata-rata **{col}**: {df[col].mean():.2f}"
    if "total" in query_lower or "jumlah" in query_lower or "sum" in query_lower:
        if match_cols:
            col = df.columns[cols.get_loc(match_cols[0])]
            return f"Σ Total **{col}**: {df[col].sum():,.0f}"
    if "max" in query_lower or "tertinggi" in query_lower:
        if match_cols:
            col = df.columns[cols.get_loc(match_cols[0])]
            return f"🔼 Nilai tertinggi **{col}**: {df[col].max()}"
    if "min" in query_lower or "terendah" in query_lower:
        if match_cols:
            col = df.columns[cols.get_loc(match_cols[0])]
            return f"🔽 Nilai terendah **{col}**: {df[col].min()}"

    return None  # biar fallback ke info umum

# ================= Tabs =================
tab1, tab2 = st.tabs(["📊 Data Analysis", "📚 RAG Advanced"])

# ====== MODE 1: Data Analysis ======
with tab1:
    uploaded_file = st.file_uploader("Upload file Excel/CSV untuk analisa data", type=["csv", "xls", "xlsx"])
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
            st.session_state.active_df = df

# ====== MODE 2: RAG Advanced ======
with tab2:
    rag_files = st.file_uploader(
        "Upload dokumen multi-format untuk RAG",
        type=["pdf", "txt", "docx", "pptx", "csv", "xls", "xlsx", "png", "jpg", "jpeg", "bmp", "gif"],
        accept_multiple_files=True,
        key="rag_uploader"
    )
    if rag_files:
        st.session_state.vectorstore = build_vectorstore(rag_files)
        if st.session_state.vectorstore:
            st.success("✅ Vectorstore berhasil dibuat")
        else:
            st.error("❌ Gagal membangun vectorstore dari file yang diupload.")

# ================= Chatbot (Global Root) =================
st.markdown("---")
st.subheader("💬 Chatbot")

# tampilkan riwayat chat
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)

user_query = st.chat_input("Tanyakan sesuatu...")
if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append(("user", user_query))

    response = None

    # ================= AUTO DETEKSI =================
    if st.session_state.active_df is not None and any(
        kw in user_query.lower()
        for kw in ["statistik", "summary", "deskripsi", "describe", "trend", "tren",
                   "waktu", "time", "sales", "penjualan", "kategori", "category",
                   "bar", "pie", "insight", "kesimpulan", "conclusion", "average", "rata", "total", "sum", "max", "min"]
    ):
        df = st.session_state.active_df
        query_lower = user_query.lower()

        # statistik dasar
        if any(kw in query_lower for kw in ["statistik", "summary", "deskripsi", "describe"]):
            st.dataframe(safe_describe(df))
            response = "📊 Statistik dasar dataset ditampilkan."

        # heatmap korelasi
        elif "heatmap" in query_lower or "korelasi" in query_lower:
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            response = "🔥 Heatmap korelasi antar kolom numerik ditampilkan."

        # tren/grafik
        elif "trend" in query_lower or "tren" in query_lower or "grafik" in query_lower:
            num_cols = df.select_dtypes(include="number").columns
            date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if date_cols:
                col_x = date_cols[0]
                col_y = num_cols[0] if len(num_cols) else None
                if col_y:
                    fig = px.line(df.sort_values(col_x), x=col_x, y=col_y, title=f"Trend {col_y} terhadap {col_x}")
                    st.plotly_chart(fig, use_container_width=True)
                    response = f"📈 Grafik tren **{col_y} vs {col_x}** ditampilkan."
            elif len(num_cols) >= 1:
                col = num_cols[0]
                fig = px.line(df, y=col, title=f"Trend dari kolom {col}")
                st.plotly_chart(fig, use_container_width=True)
                response = f"📈 Grafik trend untuk kolom **{col}** ditampilkan."
            else:
                response = "⚠️ Tidak ada kolom numerik untuk membuat grafik trend."

        # insight & kesimpulan
        elif any(kw in query_lower for kw in ["insight", "kesimpulan", "conclusion"]):
            cat_cols, num_cols = detect_data_types(df)
            insights = "🔍 Insight & Kesimpulan utama dataset:\n"
            if num_cols:
                top_corr = df[num_cols].corr().unstack().sort_values(ascending=False)
                top_corr = top_corr[top_corr < 1].dropna()
                if not top_corr.empty:
                    pair = top_corr.index[0]
                    insights += f"- Korelasi tertinggi antara **{pair[0]}** dan **{pair[1]}**: {top_corr.iloc[0]:.2f}\n"
                insights += f"- Nilai rata-rata untuk {num_cols[0]}: {df[num_cols[0]].mean():.2f}\n"
            if cat_cols:
                insights += f"- Kolom kategorikal utama: {', '.join(cat_cols[:3])}\n"
            response = insights

        # pertanyaan umum dataset
        else:
            general_answer = qa_general_dataset(df, user_query)
            if general_answer:
                response = general_answer
            else:
                response = f"🔍 Dataset punya {df.shape[0]} baris dan {df.shape[1]} kolom. Kolom: {', '.join(df.columns[:10])}"

    elif st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-small",
            task="text2text-generation"
        )
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        result = qa({"query": user_query})
        answer = result["result"]
        sources = result.get("source_documents", [])
        response = answer
        if sources:
            response += "\n\n**Sumber:**\n"
            for s in sources:
                response += f"- {s.metadata.get('source', '')}\n"

    else:
        response = "⚠️ Silakan upload dataset di **Data Analysis** atau dokumen di **RAG Advanced** dulu."

    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append(("assistant", response))
