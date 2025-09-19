# app-langchain.py

import streamlit as st  
import pandas as pd
import plotly.express as px
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    UnstructuredImageLoader,
)
import tempfile, os

# =====================
# Session State
# =====================
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# =====================
# Dummy LLM API wrapper (Google Gemini / Groq)
# =====================
def llm_invoke(prompt_text: str):
    """
    Gunakan API key dari st.secrets["GOOGLE_API_KEY"] atau st.secrets["GROQ_API_KEY"].
    Return response string dari LLM eksternal.
    """
    # Contoh mock response
    return f"💡 LLM menjawab: (simulasi) untuk prompt: {prompt_text[:100]}..."

# =====================
# Helper Functions
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

def generate_dataset_insight(df: pd.DataFrame):
    stats = safe_describe(df).reset_index().to_string()
    prompt = f"""
    Kamu adalah analis data. Berdasarkan statistik berikut:
    {stats}
    Buatkan insight utama dan kesimpulan secara akurat dan ringkas.
    """
    return llm_invoke(prompt)

def load_document(file_path, file_type):
    if file_type == ".pdf":
        return PyPDFLoader(file_path).load()
    elif file_type == ".txt":
        return TextLoader(file_path).load()
    elif file_type == ".docx":
        return Docx2txtLoader(file_path).load()
    elif file_type in [".pptx", ".ppt"]:
        return UnstructuredPowerPointLoader(file_path).load()
    elif file_type.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
        return UnstructuredImageLoader(file_path).load()
    else:
        return []

def process_rag_files(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
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
st.title("🤖📊 Chatbot Dashboard : Data Analysis & Advanced RAG")
st.set_page_config(
    page_title="Chatbot Dashboard & Advanced RAG",
    page_icon="🤖📊",
    layout="wide"
)

tab1, tab2 = st.tabs(["📈 Data Analysis", "📚 RAG Advanced"])

# ====== Data Analysis ======
with tab1:
    uploaded_file = st.file_uploader("Upload file Excel/CSV", type=["csv", "xls", "xlsx"])
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

        # Dropdown visualisasi
        st.subheader("⚙️ Pilih Kolom untuk Visualisasi")
        x_axis = st.selectbox("X Axis", df.columns)
        y_axis = st.selectbox("Y Axis", df.columns)

        if x_axis and y_axis:
            x_is_num = x_axis in numeric_cols or x_axis in datetime_cols
            y_is_num = y_axis in numeric_cols
            x_is_cat = x_axis in categorical_cols
            y_is_cat = y_axis in categorical_cols

            fig = None
            if x_is_num and y_is_num:
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter {y_axis} vs {x_axis}")
            elif x_axis in datetime_cols and y_is_num:
                fig = px.line(df, x=x_axis, y=y_axis, title=f"Trend {y_axis} vs {x_axis}")
            elif x_is_cat and y_is_num:
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"Bar {y_axis} per {x_axis}")
            elif x_is_cat and y_is_cat:
                crosstab = pd.crosstab(df[x_axis], df[y_axis])
                fig = px.imshow(crosstab, title=f"Frequency {x_axis} vs {y_axis}")

            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Chatbot Data Analysis
        st.subheader("💬 Chatbot Data Analysis")
        q = st.text_input("Tanyakan sesuatu tentang dataset")
        if q:
            if "insight" in q.lower() or "kesimpulan" in q.lower():
                st.write(generate_dataset_insight(df))
            else:
                st.write("🔍 Gunakan kata kunci statistik / tren / kategori / insight.")

# ====== RAG Advanced ======
with tab2:
    uploaded_files = st.file_uploader("Upload dokumen", type=["pdf","docx","pptx","txt","jpg","jpeg","png","bmp","gif"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.vectorstore = process_rag_files(uploaded_files)
        st.success("✅ Dokumen berhasil diproses!")

    st.subheader("💬 Chatbot RAG")
    q2 = st.text_input("Tanyakan sesuatu tentang dokumen")
    if q2 and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(q2)
        context = "\n".join([d.page_content for d in docs[:3]])
        prompt_text = f"Jawab pertanyaan berikut berdasarkan konteks:\n{context}\nPertanyaan: {q2}"
        st.write(llm_invoke(prompt_text))
