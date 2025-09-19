# app-langchain.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os, tempfile, requests

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, TextLoader

# =====================
# Session State Init
# =====================
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "uploaded_rag_files" not in st.session_state:
    st.session_state.uploaded_rag_files = []

# =====================
# LLM Setup
# =====================
def load_llm():
    try:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    except Exception:
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

llm = load_llm()

# =====================
# Helpers Data Analysis
# =====================
def df_info_text(df: pd.DataFrame) -> str:
    info = f"Baris: {df.shape[0]}, Kolom: {df.shape[1]}\n"
    info += "Kolom:\n" + ", ".join(df.columns[:30])
    if df.shape[1] > 30:
        info += " ..."
    return info

def detect_column_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    return numeric_cols, categorical_cols, datetime_cols

def safe_describe(df):
    try:
        return df.describe(include="all")
    except Exception:
        return pd.DataFrame()

def generate_dataset_insight(df: pd.DataFrame, question: str = None):
    stats = safe_describe(df).reset_index().to_string()
    question_section = f"Pertanyaan: {question}" if question else ""
    prompt_template = f"""
    Kamu adalah analis data profesional.
    Berdasarkan dataset berikut:
    {stats}
    {question_section}
    Jawaban harus relevan, akurat, jelas, dan mudah dipahami.
    Jika jawaban tidak tersedia, katakan: "Jawaban tidak tersedia dalam konteks yang diberikan."
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    return chain.invoke({}).content

# =====================
# RAG File Loader
# =====================
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")  # harus ada di .env

def load_image_ocr(file_path):
    with open(file_path, "rb") as f:
        r = requests.post(
            "https://api.ocr.space/parse/image",
            files={file_path: f},
            data={"apikey": OCR_SPACE_API_KEY, "language": "eng"}
        )
    try:
        result = r.json()
        parsed_text = ""
        for entry in result.get("ParsedResults", []):
            parsed_text += entry.get("ParsedText", "") + "\n"
        return parsed_text.strip()
    except:
        return ""

def load_document(file_path, file_type):
    file_type = file_type.lower()
    if file_type == ".pdf":
        return PyPDFLoader(file_path).load()
    elif file_type == ".txt":
        return TextLoader(file_path).load()
    elif file_type == ".docx":
        return Docx2txtLoader(file_path).load()
    elif file_type in [".pptx", ".ppt"]:
        return UnstructuredPowerPointLoader(file_path).load()
    elif file_type in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".jfif"]:
        text = load_image_ocr(file_path)
        if text.strip() == "":
            text = "[Tidak ada teks berhasil diekstrak dari gambar]"
        return [Document(page_content=text, metadata={"source": os.path.basename(file_path)})]
    else:
        return []

def process_rag_files(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        docs.extend(load_document(tmp_path, suffix))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs_split, embeddings)

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="🤖📊 Data & Document Chatbot", layout="wide")
st.title("🤖📊 Chatbot Dashboard : Data Analysis & RAG")

tab1, tab2 = st.tabs(["📈 Data Analysis", "📚 RAG Advanced"])

# ----- Data Analysis -----
with tab1:
    uploaded_file = st.file_uploader("Upload file Excel/CSV untuk analisa data", type=["csv", "xls", "xlsx"])
    df = None

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            xls = pd.ExcelFile(uploaded_file)
            st.session_state.dfs = {sheet: pd.read_excel(uploaded_file, sheet_name=sheet) for sheet in xls.sheet_names}
            selected_sheets = st.multiselect("📑 Pilih Sheet", list(st.session_state.dfs.keys()), default=list(st.session_state.dfs.keys())[:1])
            if selected_sheets:
                df_list = []
                for s in selected_sheets:
                    temp = st.session_state.dfs[s].copy()
                    temp["SheetName"] = s
                    df_list.append(temp)
                df = pd.concat(df_list, ignore_index=True)

    if df is not None:
        st.dataframe(df.head(10))
        numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)
        st.write(f"Kolom Numerik: {numeric_cols}")
        st.write(f"Kolom Kategorikal: {categorical_cols}")
        st.write(f"Kolom Datetime: {datetime_cols}")
        st.text(df_info_text(df))
        st.write(f"**Data shape:** {df.shape}")        

        st.subheader("⚙️ Pilih Kolom untuk Visualisasi")
        x_axis = st.selectbox("Kolom X Axis", df.columns)
        y_axis = st.selectbox("Kolom Y Axis", df.columns)

        if x_axis and y_axis:
            x_is_num = x_axis in numeric_cols or x_axis in datetime_cols
            y_is_num = y_axis in numeric_cols
            x_is_cat = x_axis in categorical_cols
            y_is_cat = y_axis in categorical_cols

            fig = None
            if x_is_num and y_is_num:
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"📈 Scatter {y_axis} vs {x_axis}")
            elif x_axis in datetime_cols and y_is_num:
                fig = px.line(df, x=x_axis, y=y_axis, title=f"📈 Tren {y_axis} vs {x_axis}")
            elif x_is_cat and y_is_num:
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"📊 Bar {y_axis} per {x_axis}")
            elif x_is_num and y_is_cat:
                fig = px.bar(df, x=y_axis, y=x_axis, title=f"📊 Bar {x_axis} per {y_axis}")
            elif x_is_cat and y_is_cat:
                crosstab = pd.crosstab(df[x_axis], df[y_axis])
                fig = px.imshow(crosstab, title=f"🔢 Frekuensi {x_axis} vs {y_axis}")

            if fig:
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("💬 Chatbot Data Analysis")
        q = st.text_input("Tanyakan sesuatu tentang dataset")
        if q:
            st.write(generate_dataset_insight(df, question=q))

# ----- RAG Advanced -----
with tab2:
    uploaded_files = st.file_uploader(
        "Upload dokumen PDF/DOCX/PPTX/TXT/Gambar (multi-file)",
        type=["pdf","docx","pptx","txt","jpg","jpeg","png","bmp","gif","jfif"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.session_state.uploaded_rag_files = uploaded_files
        st.markdown("**File terupload:**")
        for f in uploaded_files:
            st.write(f"• {f.name}")

    if st.button("🚀 Build Vector Store") and st.session_state.uploaded_rag_files:
        with st.spinner("Memproses dokumen dan membuat vector store..."):
            vs = process_rag_files(st.session_state.uploaded_rag_files)
            st.session_state.vectorstore = vs
            st.success(f"✅ Vectorstore terbangun. Total dokumen: {len(st.session_state.uploaded_rag_files)} | Total chunk: {len(vs.docstore._dict)}")

    st.subheader("💬 Chatbot RAG")
    q2 = st.text_input("Tanyakan sesuatu tentang dokumen")
    if q2 and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(q2)
        if docs:
            context = "\n".join([d.page_content for d in docs])
            prompt = ChatPromptTemplate.from_template(f"""
            Jawab pertanyaan berikut secara akurat, jelas, dan ringkas berdasarkan dokumen.
            Jika jawaban tidak ada, katakan: "Jawaban tidak tersedia dalam konteks yang diberikan."
            Pertanyaan: {{q}}
            Konteks: {{context}}
            Jawaban ringkas:
            """)
            chain = prompt | llm
            st.write(chain.invoke({"q": q2, "context": context}).content)
        else:
            st.warning("Tidak ada teks relevan berhasil diekstrak dari dokumen.")

