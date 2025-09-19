# app-langchain.py

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

from PyPDF2 import PdfReader
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

# ================= Session State =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Data Analysis"
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "df_active" not in st.session_state:
    st.session_state.df_active = None

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

def detect_date_columns(df):
    return [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]

def safe_describe(df):
    try:
        return df.describe(include="all")
    except Exception:
        return df.describe()

def df_info_text(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

# ================= Tabs =================
tab1, tab2 = st.tabs(["📊 Data Analysis", "📚 RAG Advanced"])

with tab1:
    st.session_state.active_tab = "Data Analysis"
    st.subheader("Upload file untuk Data Analysis")

    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"], key="data_uploader")
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state.dfs = {"CSV": df}
        else:
            xls = pd.ExcelFile(uploaded_file)
            st.session_state.dfs = {s: pd.read_excel(uploaded_file, sheet_name=s) for s in xls.sheet_names}
        st.success("✅ File berhasil diupload")

        sheet_names = list(st.session_state.dfs.keys())
        selected_sheets = st.multiselect("Pilih Sheet", sheet_names, default=sheet_names[:1])
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

            st.session_state.df_active = df
            st.dataframe(df.head(10))
            categorical_cols, numeric_cols = detect_data_types(df)
            st.write(f"Kolom Numerik: {numeric_cols}")
            st.write(f"Kolom Kategorikal: {categorical_cols}")
            st.text(df_info_text(df))
            st.write(f"**Data shape:** {df.shape}")
            st.dataframe(safe_describe(df))           

with tab2:
    st.session_state.active_tab = "RAG Advanced"
    st.subheader("Upload dokumen multi-format untuk RAG")

    rag_files = st.file_uploader(
        "Upload dokumen (PDF, TXT, DOCX, PPTX, CSV, XLSX, Gambar)",
        type=["pdf", "txt", "docx", "pptx", "csv", "xls", "xlsx", "png", "jpg", "jpeg", "bmp", "gif"],
        accept_multiple_files=True,
        key="rag_uploader"
    )
    if rag_files:
        st.session_state.vectorstore = build_vectorstore(rag_files)
        if st.session_state.vectorstore:
            st.success("✅ Vectorstore berhasil dibuat")
        else:
            st.error("❌ Gagal membuat vectorstore")

# ================= Chatbot =================
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

    # ===== Data Analysis Mode =====
    if st.session_state.active_tab == "Data Analysis":
        if st.session_state.df_active is None:
            response = "⚠️ Silakan upload file data di tab **Data Analysis** dulu."
        else:
            df = st.session_state.df_active
            query_lower = user_query.lower()

            if "statistik" in query_lower or "describe" in query_lower:
                response = "📊 Statistik ringkas ditampilkan di atas."
                st.dataframe(df.describe(include="all"))

            elif "heatmap" in query_lower or "korelasi" in query_lower:
                corr = df.corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
                response = "🔥 Heatmap korelasi antar kolom numerik ditampilkan."

            elif "trend" in query_lower or "grafik" in query_lower:
                date_cols = detect_date_columns(df)
                num_cols = df.select_dtypes(include="number").columns.tolist()

                if not date_cols and not num_cols:
                    response = "⚠️ Tidak ada kolom yang cocok untuk grafik tren."
                else:
                    st.info("📊 Silakan pilih kolom X dan Y untuk grafik tren:")
                    x_axis = st.selectbox("Kolom X:", date_cols + df.columns.tolist(), key="trend_x")
                    y_axis = st.selectbox("Kolom Y:", num_cols, key="trend_y")

                    if x_axis and y_axis:
                        trend_df = df[[x_axis, y_axis]].dropna().sort_values(x_axis)
                        if pd.api.types.is_datetime64_any_dtype(trend_df[x_axis]):
                            trend_df = trend_df.groupby(x_axis, as_index=False)[y_axis].sum()
                        fig = px.line(trend_df, x=x_axis, y=y_axis, title=f"Grafik tren {y_axis} vs {x_axis}")
                        st.plotly_chart(fig, use_container_width=True)
                        response = f"📈 Grafik tren **{y_axis} vs {x_axis}** ditampilkan."
                    else:
                        response = "⚠️ Silakan pilih kolom X dan Y."

            else:
                response = f"🔍 Dataset berisi {df.shape[0]} baris dan {df.shape[1]} kolom."

    # ===== RAG Advanced Mode =====
    elif st.session_state.active_tab == "RAG Advanced":
        if not st.session_state.vectorstore:
            response = "⚠️ Silakan upload dokumen di tab **RAG Advanced** dulu."
        else:
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

    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append(("assistant", response))
