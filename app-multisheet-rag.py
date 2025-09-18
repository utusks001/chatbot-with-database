# app-multisheet-rag.py

import streamlit as st
import io, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv, set_key

# LangChain Core
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents.agent_types import AgentType

# HuggingFace fallback
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

from utils1 import detect_data_types, recommend_and_plot

# ========== Setup Env ==========
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

IS_STREAMLIT_CLOUD = "STREAMLIT_RUNTIME" in os.environ

# LangSmith (opsional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get(
    "LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", "")
)

# ========== Sidebar Konfigurasi ==========
with st.sidebar:
    st.header("🔑 Konfigurasi API Key")

    GOOGLE_API_KEY = (
        st.session_state.get("GOOGLE_API_KEY")
        or st.secrets.get("GOOGLE_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
    )

    api_key_input = st.text_input(
        "Masukkan Google API Key:",
        type="password",
        value=GOOGLE_API_KEY,
    )

    if st.button("💾 Simpan API Key"):
        if api_key_input.strip():
            GOOGLE_API_KEY = api_key_input.strip()
            st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

            if IS_STREAMLIT_CLOUD:
                st.info("☁️ Streamlit Cloud → Gunakan *Settings → Secrets* untuk menyimpan permanen.")
            else:
                try:
                    set_key(dotenv_path, "GOOGLE_API_KEY", GOOGLE_API_KEY)
                    st.success("✅ API Key disimpan ke .env (lokal).")
                except Exception as e:
                    st.warning(f"Gagal simpan ke .env: {e}")
        else:
            st.error("API Key tidak boleh kosong!")

    if GOOGLE_API_KEY:
        st.success("GOOGLE_API_KEY berhasil dimuat ✅")
    else:
        st.warning("⚠️ GOOGLE_API_KEY belum tersedia.")

    # Chat history
    st.header("Riwayat Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ========= Helper Functions ==========
def safe_describe(df: pd.DataFrame):
    try:
        return df.describe(include="all").transpose()
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def df_info_text(df: pd.DataFrame):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

def get_embeddings():
    """Pilih embeddings dengan fallback."""
    if GOOGLE_API_KEY:
        try:
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        except Exception as e:
            st.warning(f"❌ Gagal pakai Gemini embeddings, fallback HuggingFace: {e}")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(file):
    """Bangun vectorstore dari dokumen yang diupload."""
    if file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
        docs = [Document(page_content=text)]
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        text = df.to_csv(index=False)
        docs = [Document(page_content=text)]
    elif file.name.endswith(".xlsx"):
        xls = pd.ExcelFile(file)
        text = "\n".join(
            [f"[{sheet}]\n" + pd.read_excel(file, sheet_name=sheet).to_csv(index=False) for sheet in xls.sheet_names]
        )
        docs = [Document(page_content=text)]
    else:
        docs = [Document(page_content=file.read().decode("utf-8", errors="ignore"))]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = get_embeddings()
    return FAISS.from_documents(split_docs, embeddings)

# ========= Main App ==========
st.set_page_config(page_title="DataViz + RAG Chatbot", layout="wide")
st.title("🤖 Chatbot Analisis Data + Document RAG (Gemini + HuggingFace Fallback)")

tab1, tab2 = st.tabs(["📊 Analisis Excel/CSV", "📄 RAG Document QA"])

# ======== Tab 1: Analisis Data ========
with tab1:
    uploaded_file = st.file_uploader(
        "Upload file Excel (.xls, .xlsx) atau CSV (.csv)",
        type=["csv", "xls", "xlsx"],
        key="data_upload"
    )

    if uploaded_file is not None:
        # Load data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state.dfs = {"CSV": df}
        else:
            xls = pd.ExcelFile(uploaded_file)
            st.session_state.dfs = {
                sheet: pd.read_excel(uploaded_file, sheet_name=sheet)
                for sheet in xls.sheet_names
            }

        with st.sidebar:
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
        st.write("**Info():**")
        st.text(df_info_text(df))
        st.write("**Describe():**")
        st.dataframe(safe_describe(df))

        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            st.write("**Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        agent_key = f"agent_{sheet_label}"
        if agent_key not in st.session_state and GOOGLE_API_KEY:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
                st.session_state[agent_key] = create_pandas_dataframe_agent(
                    llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS, allow_dangerous_code=True
                )
                st.success(f"Chatbot siap! (Sheet: {sheet_label})")
            except Exception as e:
                st.error(f"Gagal inisialisasi chatbot: {e}")
                st.stop()

        if GOOGLE_API_KEY:
            user_query = st.chat_input("Tanyakan sesuatu tentang data...")
            if user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)
                with st.spinner("Memproses..."):
                    try:
                        response = st.session_state[agent_key].run(user_query)
                        with st.chat_message("assistant"):
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")

# ======== Tab 2: RAG Document QA ========
with tab2:
    rag_file = st.file_uploader("Upload dokumen untuk RAG (TXT, CSV, XLSX)", type=["txt", "csv", "xlsx"], key="rag_upload")

    if rag_file is not None:
        with st.spinner("Membangun vectorstore..."):
            vectorstore = build_vectorstore(rag_file)

        if GOOGLE_API_KEY:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
        else:
            st.warning("⚠️ GOOGLE_API_KEY kosong, fallback jawaban dummy.")
            llm = None

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        if llm:
            qa = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
            query = st.text_input("Tanyakan sesuatu tentang dokumen...")
            if query:
                with st.spinner("Mencari jawaban..."):
                    result = qa.invoke(query)
                    st.write("**Jawaban:**", result["result"])
                    with st.expander("📚 Sumber"):
                        for doc in result["source_documents"]:
                            st.markdown(doc.page_content[:500] + "...")
        else:
            st.error("Tidak ada LLM aktif untuk menjawab.")
