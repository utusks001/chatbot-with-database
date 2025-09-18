# app-multisheet.py

import streamlit as st
import io
import os

# Data analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import toml
from dotenv import load_dotenv, set_key

# LangChain & LLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents.agent_types import AgentType
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ========== Load .env kalau ada ==========
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# ========== Setup LangSmith ==========
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))

# ========== Sidebar API Key ==========
with st.sidebar:
    st.header("🔑 Konfigurasi API Key")

    # Fallback urutan key
    GOOGLE_API_KEY = (
        st.session_state.get("GOOGLE_API_KEY", "")
        or st.secrets.get("GOOGLE_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
    )

    api_key_input = st.text_input(
        "Masukkan GOOGLE_API_KEY (buat baru di https://aistudio.google.com/apikey)",
        type="password",
        value=GOOGLE_API_KEY
    )

    if st.button("💾 Simpan API Key"):
        if api_key_input.strip():
            GOOGLE_API_KEY = api_key_input.strip()
            st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

            # Simpan ke .env
            set_key(dotenv_path, "GOOGLE_API_KEY", GOOGLE_API_KEY)

            # Simpan ke secrets.toml
            secrets_path = Path(".streamlit/secrets.toml")
            secrets_path.parent.mkdir(exist_ok=True)
            secrets = {}
            if secrets_path.exists():
                secrets = toml.load(secrets_path)
            secrets["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            with open(secrets_path, "w") as f:
                toml.dump(secrets, f)

            st.success("✅ API Key berhasil disimpan!")
        else:
            st.error("API Key tidak boleh kosong!")

    if GOOGLE_API_KEY:
        st.success("GOOGLE_API_KEY aktif ✅")
    else:
        st.warning("⚠️ GOOGLE_API_KEY belum diisi, chatbot tidak bisa jalan.")

    # ===== Chat history =====
    st.header("Riwayat Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ========= Helper Functions =========
def safe_describe(df: pd.DataFrame):
    try:
        return df.describe(include="all").transpose()
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def df_info_text(df: pd.DataFrame):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

# ========= Main =========
st.set_page_config(page_title="DataViz Chatbot + RAG", layout="wide")
st.title("🤖 Chatbot Otomasi Analisis Data + Document RAG")

uploaded_file = st.file_uploader(
    "Upload file Excel (.xls, .xlsx), CSV (.csv), atau TXT/MD/PDF (.txt, .md, .pdf) untuk RAG", 
    type=["csv", "xls", "xlsx", "txt", "md", "pdf"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.session_state.dfs = {"CSV": df}
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        xls = pd.ExcelFile(uploaded_file)
        st.session_state.dfs = {s: pd.read_excel(uploaded_file, sheet_name=s) for s in xls.sheet_names}
    else:
        # ===== Document RAG mode =====
        if not GOOGLE_API_KEY:
            st.error("⚠️ Masukkan GOOGLE_API_KEY dulu untuk aktifkan RAG.")
            st.stop()

        text = uploaded_file.read().decode("utf-8", errors="ignore")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_text(text)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=GOOGLE_API_KEY
        )
        vectorstore = FAISS.from_texts(docs, embeddings)
        retriever = vectorstore.as_retriever()

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
        )

        st.success("✅ RAG chatbot siap!")
        query = st.chat_input("Tanyakan sesuatu tentang dokumen...")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Memproses dokumen..."):
                try:
                    response = qa_chain.run(query)
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
        st.stop()

    # ========== Multi-sheet Excel/CSV ==========
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

    # Heatmap
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        st.write("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Inisialisasi agent
    agent_key = f"agent_{sheet_label}"
    if agent_key not in st.session_state and GOOGLE_API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY
            )
            st.session_state[agent_key] = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
            )
            st.success(f"Chatbot siap! (Sheet: {sheet_label})")
        except Exception as e:
            st.error(f"Gagal inisialisasi chatbot: {e}")
            st.stop()
    elif not GOOGLE_API_KEY:
        st.warning("⚠️ GOOGLE_API_KEY belum diisi.")

    # Chat input
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
